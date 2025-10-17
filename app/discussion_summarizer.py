import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class DiscussionSummarizer:
    """
    특정 사용자의 토론 참여를 분석하여 주제별 대표 발언과 개인화된 피드백을 생성.

    - GPT-4o-mini를 사용하여 사용자의 발언을 분석
    - 각 토론 주제별로 사용자의 대표 발언 선정
    - 사용자의 전체 발언에 대한 요약과 격려 메시지 제공
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.max_messages = int(os.getenv("DISCUSSION_SUMMARY_MAX_MESSAGES", "60"))
        self.model = os.getenv("DISCUSSION_SUMMARY_MODEL", "gpt-4o-mini")

        self.system_prompt = (
            "당신은 CJ 식음 서비스 매니저 교육 프로그램의 토론 코치입니다. "
            "특정 참여자의 발언을 세심하게 분석하여 긍정적이고 구체적인 피드백을 제공합니다. "
            "참여자가 토론에서 보여준 강점과 기여를 발견하고, 따뜻하게 격려하며 칭찬하세요. "
            "반드시 JSON 포맷으로만 응답하세요."
        )

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 OPENAI_API_KEY를 추가해주세요."
            )

        self.client = OpenAI(api_key=self.api_key)
        print("DiscussionSummarizer: OpenAI API 설정 완료")

    def summarize_user(
        self,
        user_id: str,
        chat_history: List[Dict],
        discussion_topics: List[Dict[str, Optional[str]]],
        max_statements_per_topic: int = 3,
    ) -> Dict:
        """
        특정 사용자의 토론 참여를 분석하여 주제별 대표 발언과 피드백 생성.

        Args:
            user_id: 분석할 사용자 ID (nickname 또는 user_id)
            chat_history: 전체 채팅 내역 리스트 (nickname/user_id/text/timestamp 포함)
            discussion_topics: 토론 주제 목록 [{"name": "...", "description": "..."}]
            max_statements_per_topic: 주제별 대표 발언 최대 개수 (기본값: 3)

        Returns:
            {
                "user_id": "사용자ID",
                "topics": [
                    {
                        "topic": "주제명",
                        "user_statements": [
                            {
                                "text": "발언 내용",
                                "reason": "선정 이유 (칭찬)"
                            }
                        ],
                        "feedback": "주제별 사용자 발언 요약 및 격려"
                    }
                ],
                "overall_feedback": "전체 발언에 대한 종합 피드백 및 격려",
                "generated_at": "생성 시각"
            }
        """
        # 토론 주제 정규화
        normalized_topics = self._normalize_topics(discussion_topics)

        # 전체 채팅 내역과 사용자 발언 인덱싱
        indexed_history, user_messages = self._index_chat_history(
            chat_history[-self.max_messages:] if chat_history else [],
            user_id
        )

        if not user_messages:
            return self._empty_response(user_id, normalized_topics)

        safe_max_statements = max(1, min(max_statements_per_topic, 8))

        try:
            result = self._analyze_with_gpt(
                user_id,
                indexed_history,
                user_messages,
                normalized_topics,
                safe_max_statements,
            )
            result["generated_at"] = datetime.now().isoformat()
            return result
        except Exception as e:
            print(f"DiscussionSummarizer: GPT 분석 실패 - {e}")
            raise

    # ========== GPT Analysis ==========

    def _analyze_with_gpt(
        self,
        user_id: str,
        indexed_history: List[Dict],
        user_messages: List[Dict],
        topics: List[Dict[str, Optional[str]]],
        max_statements: int,
    ) -> Dict:
        """GPT를 활용한 사용자 발언 분석 및 피드백 생성"""

        # 토론 주제 텍스트 포맷팅
        topic_text = self._format_topic_text(topics)

        # 전체 채팅 내역 (맥락 제공)
        all_conversation = json.dumps(indexed_history, ensure_ascii=False, indent=2)

        # 사용자 발언만 추출
        user_conversation = json.dumps(user_messages, ensure_ascii=False, indent=2)

        # 응답 스키마
        response_schema = json.dumps(
            {
                "user_id": "string",
                "topics": [
                    {
                        "topic": "string",
                        "user_statements": [
                            {
                                "message_id": 1,
                                "reason": "string (이 발언이 왜 훌륭한지 구체적으로 칭찬, 30-50자)"
                            }
                        ],
                        "feedback": "string (150-200자, 이 주제에서 사용자가 보여준 기여를 칭찬하고 격려)"
                    }
                ],
                "overall_feedback": "string (200-300자, 전체 토론에서 사용자가 보여준 강점을 종합적으로 칭찬하고 격려)"
            },
            ensure_ascii=False,
            indent=2,
        )

        user_prompt = (
            f"분석 대상 사용자: {user_id}\n\n"
            f"{topic_text}\n\n"
            f"=== 전체 채팅 내역 (맥락 파악용) ===\n"
            f"{all_conversation}\n\n"
            f"=== {user_id}님의 발언 목록 ===\n"
            f"{user_conversation}\n\n"
            "📋 분석 요구사항:\n\n"
            "1. 각 토론 주제별로 분석:\n"
            f"   - {user_id}님의 발언 중 해당 주제와 관련된 대표 발언을 최대 {max_statements}개 선정\n"
            "   - message_id는 위 발언 목록의 id 값만 사용\n"
            "   - reason: 왜 이 발언이 훌륭한지 구체적으로 칭찬 (30-50자)\n"
            "   - feedback: 이 주제에서 사용자가 보여준 기여와 강점을 칭찬하고 격려 (150-200자)\n"
            "     * 어떤 관점을 제시했는지\n"
            "     * 어떤 경험이나 인사이트를 공유했는지\n"
            "     * CJ의 가치(정직/열정/창의/존중) 중 어떤 부분을 잘 보여줬는지\n\n"
            "2. 전체 종합 피드백 (overall_feedback, 200-300자):\n"
            f"   - {user_id}님이 전체 토론에서 보여준 참여 태도와 강점을 종합\n"
            "   - 구체적인 발언 내용을 인용하며 칭찬\n"
            "   - 따뜻하고 진심어린 격려로 마무리\n\n"
            "3. 작성 스타일:\n"
            "   - 존댓말 사용 (~해주셨습니다, ~주신 점이)\n"
            "   - 구체적인 발언 내용 인용\n"
            "   - 긍정적이고 교육적인 톤\n"
            "   - 비판 절대 금지, 모든 기여를 긍정적으로 해석\n\n"
            "4. ⚠️ 중요:\n"
            "   - user_statements의 message_id는 반드시 위 발언 목록의 id 값만 사용\n"
            "   - JSON 형식만 출력, 다른 텍스트 절대 금지\n\n"
            "응답 JSON 스키마:\n"
            f"{response_schema}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )

        parsed = json.loads(response.choices[0].message.content)

        # message_id를 실제 발언 내용으로 변환
        return self._resolve_result(parsed, user_messages, topics)

    def _resolve_result(
        self,
        gpt_output: Dict,
        user_messages: List[Dict],
        topics: List[Dict[str, Optional[str]]],
    ) -> Dict:
        """GPT 응답의 message_id를 실제 발언 텍스트로 변환"""

        # message_id -> 메시지 내용 매핑
        message_map = {msg["id"]: msg for msg in user_messages}
        message_map.update({str(msg["id"]): msg for msg in user_messages})

        resolved_topics = []

        for topic_data in gpt_output.get("topics", []):
            topic_name = topic_data.get("topic", "").strip()

            # 대표 발언 변환
            statements = []
            for stmt in topic_data.get("user_statements", []):
                msg_id = stmt.get("message_id")

                # message_id로 실제 메시지 찾기
                lookup_key = msg_id if msg_id in message_map else str(msg_id)
                msg = message_map.get(lookup_key)

                if msg:
                    statements.append({
                        "text": msg["text"],
                        "reason": stmt.get("reason", "").strip() or "주제와 관련된 중요한 발언입니다."
                    })

            resolved_topics.append({
                "topic": topic_name,
                "user_statements": statements,
                "feedback": topic_data.get("feedback", "").strip() or "참여해주셔서 감사합니다."
            })

        # 원래 주제 순서에 맞춰 정렬
        ordered_topics = []
        for original_topic in topics:
            found = False
            for resolved in resolved_topics:
                if resolved["topic"].lower().strip() == original_topic["name"].lower().strip():
                    ordered_topics.append(resolved)
                    found = True
                    break

            if not found:
                ordered_topics.append({
                    "topic": original_topic["name"],
                    "user_statements": [],
                    "feedback": "이 주제에 대한 발언이 없었습니다."
                })

        return {
            "user_id": gpt_output.get("user_id", ""),
            "topics": ordered_topics,
            "overall_feedback": gpt_output.get("overall_feedback", "").strip() or "토론에 참여해주셔서 감사합니다."
        }

    # ========== Helper Methods ==========

    def _normalize_topics(
        self, topics: List[Dict[str, Optional[str]]]
    ) -> List[Dict[str, Optional[str]]]:
        """토론 주제를 name/description 구조로 정규화"""
        normalized = []
        for item in topics:
            if isinstance(item, dict):
                name = item.get("name") or item.get("topic") or item.get("title")
                description = item.get("description") or item.get("detail")
            else:
                name = str(item)
                description = None

            if name:
                normalized.append({"name": name.strip(), "description": description})

        return normalized

    def _index_chat_history(
        self, chat_history: List[Dict], target_user_id: str
    ) -> tuple[List[Dict], List[Dict]]:
        """
        채팅 내역을 인덱싱하고 특정 사용자의 발언만 추출.

        Returns:
            (전체_인덱싱된_내역, 사용자_발언_목록)
        """
        indexed_all = []
        user_messages = []

        for idx, message in enumerate(chat_history, 1):
            speaker = (
                message.get("nickname")
                or message.get("user_id")
                or message.get("speaker")
                or "참여자"
            )
            text = (message.get("text") or "").strip()
            timestamp = message.get("timestamp")

            if not text:
                continue

            entry = {
                "id": idx,
                "speaker": speaker,
                "text": text,
            }
            if timestamp:
                entry["timestamp"] = timestamp

            indexed_all.append(entry)

            # 사용자의 발언만 별도 수집
            if speaker == target_user_id or message.get("user_id") == target_user_id:
                user_messages.append(entry)

        return indexed_all, user_messages

    def _format_topic_text(self, topics: List[Dict[str, Optional[str]]]) -> str:
        """GPT 프롬프트용 토론 주제 텍스트"""
        lines = ["토론 주제 목록:"]
        for idx, topic in enumerate(topics, 1):
            description = topic.get("description")
            if description:
                lines.append(f"{idx}. {topic['name']} - {description}")
            else:
                lines.append(f"{idx}. {topic['name']}")
        return "\n".join(lines)

    def _empty_response(
        self, user_id: str, topics: List[Dict[str, Optional[str]]]
    ) -> Dict:
        """사용자 발언이 없을 때의 기본 응답"""
        return {
            "user_id": user_id,
            "topics": [
                {
                    "topic": topic["name"],
                    "user_statements": [],
                    "feedback": "이 주제에 대한 발언이 없었습니다."
                }
                for topic in topics
            ],
            "overall_feedback": "토론에 참여한 발언이 확인되지 않았습니다.",
            "generated_at": datetime.now().isoformat(),
        }
