import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class DiscussionSummarizer:
    """
    채팅 내역과 토론 주제를 분석해 대표 발언과 요약을 생성하는 유틸리티.

    - OpenAI API가 설정되어 있으면 GPT 기반 요약을 시도
    - API 키가 없거나 실패할 경우 룰 기반 요약으로 폴백
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.max_messages = int(os.getenv("DISCUSSION_SUMMARY_MAX_MESSAGES", "60"))
        self.model = os.getenv("DISCUSSION_SUMMARY_MODEL", "gpt-4o-mini")

        self.system_prompt = (
            "당신은 CJ 식음 서비스 매니저 교육 프로그램의 토론 분석가입니다. "
            "반드시 JSON 포맷으로만 응답하며, 제공된 message_id 목록에서만 대표 발언을 선택하세요."
        )

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.gpt_enabled = True
            print("DiscussionSummarizer: OpenAI API 키 설정 완료")
        else:
            self.client = None
            self.gpt_enabled = False
            print("DiscussionSummarizer: OpenAI API 키 미설정 - 기본 요약 사용")

    def summarize(
        self,
        chat_history: List[Dict],
        discussion_topics: List[Dict[str, Optional[str]]],
        focus_user: Optional[str] = None,
        max_statements_per_topic: int = 3,
    ) -> Dict:
        """
        대표 발언과 토론 요약 생성.

        Args:
            chat_history: 채팅 내역 리스트 (nickname/user_id/text/timestamp 등 포함 가능)
            discussion_topics: 분석할 토론 주제 목록 [{"name": "...", "description": "..."}]
            focus_user: 특정 참여자 하이라이트 (선택)
            max_statements_per_topic: 주제별 대표 발언 최대 개수

        Returns:
            요약 정보 딕셔너리 (overall_summary, topics, focus_user_highlights, summary_method, generated_at)
        """
        normalized_topics = self._normalize_topics(discussion_topics)
        indexed_history, message_lookup = self._index_chat_history(
            chat_history[-self.max_messages :] if chat_history else []
        )

        if not indexed_history:
            return self._empty_response(normalized_topics)

        safe_max_statements = max(1, min(max_statements_per_topic, 8))

        if self.gpt_enabled:
            gpt_result = self._summarize_with_gpt(
                indexed_history,
                normalized_topics,
                message_lookup,
                focus_user,
                safe_max_statements,
            )
            if gpt_result:
                gpt_result["summary_method"] = "GPT 기반 요약"
                gpt_result["generated_at"] = datetime.now().isoformat()
                return gpt_result

        fallback = self._summarize_fallback(
            indexed_history,
            normalized_topics,
            focus_user,
            safe_max_statements,
        )
        fallback["summary_method"] = "룰 기반 요약"
        fallback["generated_at"] = datetime.now().isoformat()
        return fallback

    # ========== GPT Summarization ==========

    def _summarize_with_gpt(
        self,
        indexed_history: List[Dict],
        topics: List[Dict[str, Optional[str]]],
        message_lookup: Dict[Union[int, str], Dict],
        focus_user: Optional[str],
        max_statements: int,
    ) -> Optional[Dict]:
        """OpenAI GPT를 활용한 대표 발언/요약 생성"""
        try:
            topic_text = self._format_topic_text(topics)
            conversation_payload = json.dumps(
                indexed_history, ensure_ascii=False, indent=2
            )
            response_schema = json.dumps(
                {
                    "overall_summary": "string",
                    "topic_summaries": [
                        {
                            "topic": "string",
                            "summary": "string",
                            "representative_statements": [
                                {"message_id": 1, "reason": "string"}
                            ],
                        }
                    ],
                    "focus_user_highlights": [1],
                },
                ensure_ascii=False,
                indent=2,
            )

            focus_rule = (
                f"6. 특정 참여자 '{focus_user}'와 관련된 핵심 발언을 "
                "focus_user_highlights 배열에 message_id로 기록"
                if focus_user
                else "6. 가능한 한 다양한 참여자의 발언을 대표 발언에 포함"
            )

            user_prompt = (
                f"{topic_text}\n\n"
                "채팅 메시지 목록(JSON):\n"
                f"{conversation_payload}\n\n"
                "요구 사항:\n"
                "1. 전체 토론을 200자 내외 한 문단으로 요약\n"
                "2. 각 토론 주제별 핵심 요지를 120자 내외로 요약\n"
                f"3. 각 주제별 대표 발언을 최대 {max_statements}개 선정 (message_id 기반)\n"
                "4. 대표 발언은 제공된 메시지에서만 선택하고 text를 그대로 사용 (공백만 정리 가능)\n"
                "5. 대표 발언을 선택한 이유를 한 문장으로 작성해 reason 필드에 기록\n"
                f"{focus_rule}\n"
                "7. focus_user_highlights는 message_id 정수 배열로 작성 (조건 불충족 시 빈 배열)\n"
                "8. JSON 이외의 텍스트는 절대 출력하지 말 것\n\n"
                "응답 JSON 스키마 예시:\n"
                f"{response_schema}"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=900,
                response_format={"type": "json_object"},
            )

            parsed = json.loads(response.choices[0].message.content)
            return self._assemble_result(
                parsed, topics, message_lookup, max_statements
            )
        except Exception as exc:
            print(f"DiscussionSummarizer: GPT 요약 실패 - {exc}")
            return None

    def _assemble_result(
        self,
        gpt_output: Dict,
        topics: List[Dict[str, Optional[str]]],
        message_lookup: Dict[Union[int, str], Dict],
        max_statements: int,
    ) -> Dict:
        """GPT 응답을 최종 API 응답 형태로 정리"""
        topic_map: Dict[str, Dict] = {}
        for item in gpt_output.get("topic_summaries", []):
            topic_name = (item.get("topic") or "").strip()
            if not topic_name:
                continue

            summary_text = (item.get("summary") or "").strip()
            statements = self._resolve_statements(
                item.get("representative_statements"),
                message_lookup,
                max_statements,
            )

            topic_map[topic_name] = {
                "topic": topic_name,
                "summary": summary_text
                or "관련 발언이 충분하지 않아 요약을 구성하지 못했습니다.",
                "representative_statements": statements,
            }

        ordered_topics = []
        for topic in topics:
            name = topic["name"]
            matched = topic_map.get(name) or self._find_topic_match(topic_map, name)
            if matched:
                ordered_topics.append(matched)
            else:
                ordered_topics.append(
                    {
                        "topic": name,
                        "summary": "관련 발언이 충분하지 않아 요약을 구성하지 못했습니다.",
                        "representative_statements": [],
                    }
                )

        overall_summary = (
            gpt_output.get("overall_summary") or "토론 요약을 생성하지 못했습니다."
        ).strip()
        focus_entries = self._resolve_focus_highlights(
            gpt_output.get("focus_user_highlights"),
            message_lookup,
        )

        return {
            "overall_summary": overall_summary,
            "topics": ordered_topics,
            "focus_user_highlights": focus_entries,
        }

    def _resolve_statements(
        self,
        statements_payload: Optional[List],
        message_lookup: Dict[Union[int, str], Dict],
        max_statements: int,
    ) -> List[Dict[str, str]]:
        """GPT가 반환한 대표 발언을 실제 채팅과 매핑"""
        if not isinstance(statements_payload, list):
            return []

        normalized: List[Dict[str, str]] = []
        seen_ids = set()

        for raw_statement in statements_payload:
            if len(normalized) >= max_statements:
                break

            if isinstance(raw_statement, dict):
                message_id = raw_statement.get("message_id")
                reason = (raw_statement.get("reason") or "").strip()
            else:
                message_id = raw_statement
                reason = ""

            lookup_key = message_id
            if lookup_key not in message_lookup:
                lookup_key = str(message_id)

            reference = message_lookup.get(lookup_key)
            if not reference:
                continue

            if reference["id"] in seen_ids:
                continue
            seen_ids.add(reference["id"])

            normalized.append(
                {
                    "speaker": reference["speaker"],
                    "text": reference["text"],
                    "reason": reason
                    or "주제와 직접적으로 연결되는 핵심 발언으로 선정되었습니다.",
                }
            )

        return normalized

    def _resolve_focus_highlights(
        self,
        focus_payload,
        message_lookup: Dict[Union[int, str], Dict],
    ) -> Optional[List[str]]:
        """focus_user_highlights 배열을 채팅 문자열 리스트로 변환"""
        if not focus_payload:
            return None

        highlights: List[str] = []
        seen = set()

        for item in focus_payload:
            message_id = None
            reason = ""

            if isinstance(item, dict):
                message_id = item.get("message_id")
                reason = (item.get("reason") or "").strip()
            else:
                message_id = item

            lookup_key = message_id
            if lookup_key not in message_lookup:
                lookup_key = str(message_id)

            reference = message_lookup.get(lookup_key)
            if not reference:
                continue

            if reference["id"] in seen:
                continue
            seen.add(reference["id"])

            highlight = f"{reference['speaker']}: {reference['text']}"
            if reason:
                highlight += f" ({reason})"
            highlights.append(highlight)

        return highlights or None

    # ========== Fallback Summarization ==========

    def _summarize_fallback(
        self,
        indexed_history: List[Dict],
        topics: List[Dict[str, Optional[str]]],
        focus_user: Optional[str],
        max_statements: int,
    ) -> Dict:
        """룰 기반 대표 발언/요약 생성"""
        topic_results = []
        focus_highlights: List[str] = []

        for topic in topics:
            matches = self._extract_representative_statements(
                indexed_history,
                topic["name"],
                topic.get("description"),
                focus_user,
            )

            statements = matches[:max_statements]
            summary_text = self._build_rule_based_summary(topic["name"], statements)

            topic_results.append(
                {
                    "topic": topic["name"],
                    "summary": summary_text,
                    "representative_statements": [
                        self._strip_internal_fields(stmt) for stmt in statements
                    ],
                }
            )

            if focus_user:
                for stmt in statements:
                    if stmt["speaker"] == focus_user:
                        formatted = self._format_highlight(stmt)
                        if formatted and formatted not in focus_highlights:
                            focus_highlights.append(formatted)

        overall_summary = self._build_overall_summary(topic_results)

        return {
            "overall_summary": overall_summary,
            "topics": topic_results,
            "focus_user_highlights": focus_highlights or None,
        }

    def _extract_representative_statements(
        self,
        indexed_history: List[Dict],
        topic_name: str,
        topic_description: Optional[str],
        focus_user: Optional[str],
    ) -> List[Dict[str, str]]:
        """토픽 키워드 일치 정도를 기반으로 대표 발언 후보 선정"""
        keywords = self._extract_keywords(topic_name, topic_description)
        scored_statements: List[Tuple[float, Dict[str, str]]] = []

        for message in indexed_history:
            text = message["text"]
            if not text:
                continue

            speaker = message["speaker"]
            preference = 1.0 if focus_user and speaker == focus_user else 0.0
            score = self._compute_keyword_overlap(text, keywords) + preference

            if score > 0:
                scored_statements.append(
                    (
                        score,
                        {
                            "message_id": message["id"],
                            "speaker": speaker,
                            "text": text,
                            "reason": self._build_reason(text, keywords),
                        },
                    )
                )

        scored_statements.sort(key=lambda item: item[0], reverse=True)

        if not scored_statements:
            recent_candidates = [
                {
                    "message_id": message["id"],
                    "speaker": message["speaker"],
                    "text": message["text"],
                    "reason": "토픽과 직접적인 연관 표현은 없지만 전체 맥락에 기여한 발언입니다.",
                }
                for message in indexed_history[-5:]
                if message["text"]
            ]
            return list(reversed(recent_candidates))

        return [item[1] for item in scored_statements]

    # ========== Shared Utilities ==========

    def _normalize_topics(
        self, topics: List[Dict[str, Optional[str]]]
    ) -> List[Dict[str, Optional[str]]]:
        """토픽 입력을 name/description 구조로 정규화"""
        normalized = []
        for item in topics:
            if isinstance(item, dict):
                name = item.get("name") or item.get("topic") or item.get("title")
                description = item.get("description") or item.get("detail")
            else:
                name = str(item)
                description = None

            if not name:
                continue

            normalized.append({"name": name.strip(), "description": description})

        return normalized

    def _index_chat_history(
        self, chat_history: List[Dict]
    ) -> Tuple[List[Dict], Dict[Union[int, str], Dict]]:
        """채팅 내역에 연속 ID를 부여하고 조회용 맵을 생성"""
        indexed = []
        lookup: Dict[Union[int, str], Dict] = {}

        for idx, message in enumerate(chat_history, 1):
            speaker = (
                message.get("nickname")
                or message.get("user_id")
                or message.get("speaker")
                or "참여자"
            )
            text = (message.get("text") or "").strip()
            timestamp = message.get("timestamp")

            entry = {
                "id": idx,
                "speaker": speaker,
                "text": text,
            }
            if timestamp:
                entry["timestamp"] = timestamp

            indexed.append(entry)
            lookup[idx] = entry
            lookup[str(idx)] = entry

        return indexed, lookup

    def _format_topic_text(
        self, topics: List[Dict[str, Optional[str]]]
    ) -> str:
        """GPT 프롬프트용 토픽 정보 문자열"""
        lines = ["토론 주제 목록:"]
        for idx, topic in enumerate(topics, 1):
            description = topic.get("description")
            if description:
                lines.append(f"{idx}. {topic['name']} - {description}")
            else:
                lines.append(f"{idx}. {topic['name']}")
        return "\n".join(lines)

    def _find_topic_match(
        self, topic_map: Dict[str, Dict], name: str
    ) -> Optional[Dict]:
        """대소문자/공백 차이를 허용해 토픽을 찾는다"""
        normalized_name = name.lower().strip()
        for key, value in topic_map.items():
            if key.lower().strip() == normalized_name:
                return value
        return None

    def _compute_keyword_overlap(self, text: str, keywords: List[str]) -> float:
        """간단한 키워드 겹침 점수"""
        lower_text = text.lower()
        overlap = sum(1 for kw in keywords if kw in lower_text)
        return float(overlap)

    def _extract_keywords(
        self, topic_name: str, topic_description: Optional[str]
    ) -> List[str]:
        """토픽 이름과 설명에서 키워드를 추출"""
        base_text = f"{topic_name} {topic_description or ''}"
        tokens = [
            token.lower()
            for token in base_text.replace(",", " ").split()
            if len(token) >= 2
        ]
        return list(dict.fromkeys(tokens))

    def _build_reason(self, text: str, keywords: List[str]) -> str:
        """대표 발언 선정 이유 생성"""
        matched = [kw for kw in keywords if kw and kw in text.lower()]
        if matched:
            matched_text = ", ".join(sorted(set(matched)))
            return f"'{matched_text}' 키워드를 언급하며 주제에 기여한 발언입니다."
        return "주제와 관련된 경험이나 의견을 구체적으로 공유한 발언입니다."

    def _strip_internal_fields(self, statement: Dict[str, str]) -> Dict[str, str]:
        """API 응답 모델에 맞게 내부 필드를 정리"""
        return {
            "speaker": statement["speaker"],
            "text": statement["text"],
            "reason": statement.get("reason"),
        }

    def _format_highlight(self, statement: Dict[str, str]) -> Optional[str]:
        """focus_user 하이라이트용 문자열 생성"""
        text = statement.get("text")
        if not text:
            return None

        highlight = f"{statement['speaker']}: {text}"
        reason = statement.get("reason")
        if reason:
            highlight += f" ({reason})"
        return highlight

    def _build_rule_based_summary(
        self, topic_name: str, statements: List[Dict[str, str]]
    ) -> str:
        """간단한 토픽 요약 문장 생성"""
        if not statements:
            return f"{topic_name} 주제에서는 뚜렷한 대표 발언이 확인되지 않았습니다."

        speakers = sorted({stmt["speaker"] for stmt in statements})
        speaker_text = ", ".join(speakers)
        return (
            f"{topic_name}에 대해 {speaker_text} 등이 "
            f"실제 경험과 의견을 바탕으로 논의를 이끌었습니다."
        )

    def _build_overall_summary(self, topic_results: List[Dict]) -> str:
        """토픽 요약을 바탕으로 전체 요약 문장 생성"""
        participating_speakers = defaultdict(int)
        for result in topic_results:
            for stmt in result["representative_statements"]:
                participating_speakers[stmt["speaker"]] += 1

        if not participating_speakers:
            return "채팅 내역이 충분하지 않아 전체 요약을 생성하지 못했습니다."

        top_speakers = sorted(
            participating_speakers.items(), key=lambda item: item[1], reverse=True
        )
        highlighted = ", ".join(
            [f"{name}({count}회)" for name, count in top_speakers[:3]]
        )
        return (
            f"전체적으로 {highlighted}가 핵심 발언을 주도하며 주제별 논의를 진행했습니다."
        )

    def _empty_response(self, topics: List[Dict[str, Optional[str]]]) -> Dict:
        """채팅 내역이 없을 때의 기본 응답"""
        return {
            "overall_summary": "채팅 내역이 없어 요약을 생성하지 못했습니다.",
            "topics": [
                {
                    "topic": topic["name"],
                    "summary": "채팅 내역이 없어 요약을 생성하지 못했습니다.",
                    "representative_statements": [],
                }
                for topic in topics
            ],
            "focus_user_highlights": None,
            "summary_method": "데이터 없음",
            "generated_at": datetime.now().isoformat(),
        }
