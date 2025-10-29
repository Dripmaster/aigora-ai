import json
import os
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()


class DiscussionSummarizer:
    """
    특정 사용자의 토론 참여를 분석하여 주제별 관련 발언 요약을 생성.

    - GPT-4o-mini를 사용하여 사용자의 발언을 분석
    - 각 토론 주제별로 연관성 점수 산출 (0~1)
    - 주제와 관련있는 발언만 모아서 요약문 생성
    - 관련 발언이 없는 경우 명시적으로 표시
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.max_messages = int(os.getenv("DISCUSSION_SUMMARY_MAX_MESSAGES", "60"))
        self.model = os.getenv("DISCUSSION_SUMMARY_MODEL", "gpt-4o-mini")

        self.fast_mode = os.getenv("DISCUSSION_SUMMARY_FAST", "1") == "1"
        self.cache_ttl = int(os.getenv("DISCUSSION_SUMMARY_CACHE_TTL", "300"))  # seconds
        self.cache_max = int(os.getenv("DISCUSSION_SUMMARY_CACHE_MAX", "256"))
        self._cache: dict[str, tuple[float, dict]] = {}

        self.system_prompt = (
            "당신은 CJ 식음 서비스 매니저 교육 프로그램의 토론 분석가입니다. "
            "특정 참여자의 발언을 객관적으로 분석하여 주제별로 관련 발언을 요약합니다. "
            "발언의 핵심 내용을 간결하고 명확하게 정리하세요. "
            "반드시 JSON 포맷으로만 응답하세요."
        )

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 OPENAI_API_KEY를 추가해주세요."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        print("DiscussionSummarizer: OpenAI API 설정 완료")

    def _mk_key(self, user_id: str, chat_history: List[Dict], discussion_topics: List[Dict]) -> str:
        blob = json.dumps({"u": user_id, "h": chat_history, "t": discussion_topics}, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _get_cache(self, key: str) -> Optional[Dict]:
        now = time.time()
        hit = self._cache.get(key)
        if not hit:
            return None
        ts, value = hit
        if now - ts > self.cache_ttl:
            # expired
            try:
                del self._cache[key]
            except KeyError:
                pass
            return None
        return value

    def _set_cache(self, key: str, value: Dict):
        # LRU-ish simple bound
        if len(self._cache) >= self.cache_max:
            # drop oldest
            oldest_key = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
            self._cache.pop(oldest_key, None)
        self._cache[key] = (time.time(), value)

    def summarize_user(
        self,
        user_id: str,
        chat_history: List[Dict],
        discussion_topics: List[Dict[str, Optional[str]]],
    ) -> Dict:
        """
        특정 사용자의 토론 참여를 분석하여 주제별 관련 발언 요약 생성.

        Args:
            user_id: 분석할 사용자 ID (nickname 또는 user_id)
            chat_history: 전체 채팅 내역 리스트 (nickname/user_id/text/timestamp 포함)
            discussion_topics: 토론 주제 목록 [{"name": "...", "description": "..."}]

        Returns:
            {
                "user_id": "사용자ID",
                "topics": [
                    {
                        "topic": "주제명",
                        "relevance_score": 0.85,  # 0~1 사이 연관성 점수
                        "related_statements": ["발언1", "발언2", ...],  # 관련있는 발언만
                        "summary": "주제 관련 발언 요약문"  # 관련 발언이 있을 때만 생성
                    }
                ],
                "generated_at": "생성 시각"
            }
        """
        # 토론 주제 정규화
        normalized_topics = self._normalize_topics(discussion_topics)
        cache_key = self._mk_key(user_id, chat_history[-self.max_messages:] if chat_history else [], normalized_topics)
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        # 전체 채팅 내역과 사용자 발언 인덱싱
        indexed_history, user_messages = self._index_chat_history(
            chat_history[-self.max_messages:] if chat_history else [],
            user_id
        )

        if not user_messages:
            empty = self._empty_response(user_id, normalized_topics)
            self._set_cache(cache_key, empty)
            return empty

        try:
            result = self._analyze_with_gpt(
                user_id,
                indexed_history,
                user_messages,
                normalized_topics,
            )
            result["generated_at"] = datetime.now().isoformat()
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            print(f"DiscussionSummarizer: GPT 분석 실패 - {e}")
            raise

    async def summarize_user_async(self, user_id: str, chat_history: List[Dict], discussion_topics: List[Dict[str, Optional[str]]]) -> Dict:
        normalized_topics = self._normalize_topics(discussion_topics)
        cache_key = self._mk_key(user_id, chat_history[-self.max_messages:] if chat_history else [], normalized_topics)
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        indexed_history, user_messages = self._index_chat_history(
            chat_history[-self.max_messages:] if chat_history else [],
            user_id
        )
        if not user_messages:
            empty = self._empty_response(user_id, normalized_topics)
            self._set_cache(cache_key, empty)
            return empty
        try:
            result = await self._analyze_with_gpt_async(
                user_id,
                indexed_history,
                user_messages,
                normalized_topics,
            )
            result["generated_at"] = datetime.now().isoformat()
            self._set_cache(cache_key, result)
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
    ) -> Dict:
        """GPT를 활용한 사용자 발언 분석 및 주제별 요약 생성"""

        # 토론 주제 텍스트 포맷팅
        topic_text = self._format_topic_text(topics)

        # 전체 채팅 내역 (맥락 제공)
        all_conversation = "[]" if self.fast_mode else json.dumps(indexed_history, ensure_ascii=False, separators=(",", ":"))

        # 사용자 발언만 추출
        user_conversation = json.dumps(user_messages, ensure_ascii=False, separators=(",", ":"))

        # 응답 스키마
        response_schema = json.dumps(
            {
                "user_id": "string",
                "topics": [
                    {
                        "topic": "string (주제 이름만 정확히, 설명 포함 금지)",
                        "relevance_score": 0.85,
                        "related_message_ids": [1, 3, 5],
                        "summary": "string (100-200자, 관련 발언의 핵심 내용을 객관적으로 요약. 관련 발언이 없으면 빈 문자열)"
                    }
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # Prebuild context block to avoid backslashes in f-string expressions
        ctx_block = ""
        if not self.fast_mode:
            ctx_block = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "전체 대화 (맥락 이해용)\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{all_conversation}\n\n"
            )

        user_prompt = (
            "당신은 토론 내용을 깊이 이해하는 분석 전문가입니다.\n"
            f"{user_id}님의 발언을 읽고, 각 토론 주제와의 연관성을 스스로 판단하세요.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "토론 주제 목록\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{topic_text}\n\n"
            f"{ctx_block}"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{user_id}님의 발언\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{user_conversation}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "작업 지시\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "각 토론 주제에 대해 다음을 수행하세요:\n\n"
            "【1단계】 발언 내용 이해 및 연관성 판단\n\n"
            "   ① 주제의 이름과 설명을 읽고 핵심 의미를 파악하세요\n"
            "   ② 사용자의 각 발언을 읽고 내용과 맥락을 이해하세요\n"
            "   ③ 발언이 주제와 의미적으로 연관되는지 스스로 판단하세요\n\n"
            "   판단 원칙:\n"
            "   • 주제의 본질적 의미와 관련되면 연관성 있음\n"
            "   • 표면적 단어 일치가 아닌 내용의 의미로 판단\n"
            "   • 직접적 언급과 간접적 연관 모두 인정\n"
            "   • 발언의 의도와 맥락 고려\n"
            "   • 명백한 잡담(날씨, 인사)만 제외\n\n"
            "【2단계】 관련 발언 수집 (related_message_ids)\n\n"
            "   • 주제와 연관된다고 판단한 발언의 id를 모두 수집\n"
            "   • 위 발언 목록의 id 값만 사용\n"
            "   • 연관성이 없으면 빈 배열 []\n\n"
            "【3단계】 연관성 점수 산출 (relevance_score)\n\n"
            "   • 수집된 발언의 수와 연관 강도를 고려하여 0~1 점수 부여\n"
            "   • 발언이 많고 연관성이 강할수록 높은 점수\n"
            "   • 연관 발언이 없으면 0.0\n\n"
            "【4단계】 요약문 작성 (summary)\n\n"
            "   • related_message_ids가 비어있지 않으면 반드시 요약 작성\n"
            "   • 수집된 발언들의 핵심 내용을 객관적으로 정리 (100-200자)\n"
            "   • 평서문 형식: \"~를 언급했다\", \"~를 강조했다\", \"~를 제안했다\"\n"
            "   • 감정적 표현이나 칭찬 금지 (훌륭, 대단, 감사 등)\n"
            "   • related_message_ids가 비어있으면 빈 문자열 \"\"\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "중요 사항\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✓ 모든 주제에 대해 반드시 분석 수행\n"
            "✓ topic 필드에는 주제 이름만 정확히 입력 (설명 포함 금지)\n"
            "✓ 주제의 의미를 깊이 이해하고 넓게 해석\n"
            "✓ 발언의 표면이 아닌 내용의 본질로 판단\n"
            "✓ 관련 발언이 있으면 반드시 요약 작성\n"
            "✓ JSON 형식만 출력, 다른 텍스트 금지\n\n"
            "응답 JSON 스키마:\n"
            f"{response_schema}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        raw_response = response.choices[0].message.content
        print(f"\n{'='*80}")
        print("DEBUG: GPT 원본 응답")
        print(f"{'='*80}")
        print(raw_response)
        print(f"{'='*80}\n")

        parsed = json.loads(raw_response)

        # message_id를 실제 발언 내용으로 변환
        return self._resolve_result(parsed, user_messages, topics)

    async def _analyze_with_gpt_async(
        self,
        user_id: str,
        indexed_history: List[Dict],
        user_messages: List[Dict],
        topics: List[Dict[str, Optional[str]]],
    ) -> Dict:
        topic_text = self._format_topic_text(topics)
        all_conversation = "[]" if self.fast_mode else json.dumps(indexed_history, ensure_ascii=False, separators=(",", ":"))
        user_conversation = json.dumps(user_messages, ensure_ascii=False, separators=(",", ":"))
        response_schema = json.dumps(
            {
                "user_id": "string",
                "topics": [
                    {
                        "topic": "string (주제 이름만 정확히, 설명 포함 금지)",
                        "relevance_score": 0.85,
                        "related_message_ids": [1, 3, 5],
                        "summary": "string (100-200자, 관련 발언의 핵심 내용을 객관적으로 요약. 관련 발언이 없으면 빈 문자열)"
                    }
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        # Prebuild context block to avoid backslashes in f-string expressions
        ctx_block = ""
        if not self.fast_mode:
            ctx_block = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "전체 대화 (맥락 이해용)\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{all_conversation}\n\n"
            )
        user_prompt = (
            "당신은 토론 내용을 깊이 이해하는 분석 전문가입니다.\n"
            f"{user_id}님의 발언을 읽고, 각 토론 주제와의 연관성을 스스로 판단하세요.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "토론 주제 목록\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{topic_text}\n\n"
            f"{ctx_block}"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{user_id}님의 발언\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{user_conversation}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "작업 지시\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "각 토론 주제에 대해 다음을 수행하세요:\n\n"
            "【1단계】 발언 내용 이해 및 연관성 판단\n\n"
            "   ① 주제의 이름과 설명을 읽고 핵심 의미를 파악하세요\n"
            "   ② 사용자의 각 발언을 읽고 내용과 맥락을 이해하세요\n"
            "   ③ 발언이 주제와 의미적으로 연관되는지 스스로 판단하세요\n\n"
            "   판단 원칙:\n"
            "   • 주제의 본질적 의미와 관련되면 연관성 있음\n"
            "   • 표면적 단어 일치가 아닌 내용의 의미로 판단\n"
            "   • 직접적 언급과 간접적 연관 모두 인정\n"
            "   • 발언의 의도와 맥락 고려\n"
            "   • 명백한 잡담(날씨, 인사)만 제외\n\n"
            "【2단계】 관련 발언 수집 (related_message_ids)\n\n"
            "   • 주제와 연관된다고 판단한 발언의 id를 모두 수집\n"
            "   • 위 발언 목록의 id 값만 사용\n"
            "   • 연관성이 없으면 빈 배열 []\n\n"
            "【3단계】 연관성 점수 산출 (relevance_score)\n\n"
            "   • 수집된 발언의 수와 연관 강도를 고려하여 0~1 점수 부여\n"
            "   • 발언이 많고 연관성이 강할수록 높은 점수\n"
            "   • 연관 발언이 없으면 0.0\n\n"
            "【4단계】 요약문 작성 (summary)\n\n"
            "   • related_message_ids가 비어있지 않으면 반드시 요약 작성\n"
            "   • 수집된 발언들의 핵심 내용을 객관적으로 정리 (100-200자)\n"
            "   • 평서문 형식: \"~를 언급했다\", \"~를 강조했다\", \"~를 제안했다\"\n"
            "   • 감정적 표현이나 칭찬 금지 (훌륭, 대단, 감사 등)\n"
            "   • related_message_ids가 비어있으면 빈 문자열 \"\"\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "중요 사항\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✓ 모든 주제에 대해 반드시 분석 수행\n"
            "✓ topic 필드에는 주제 이름만 정확히 입력 (설명 포함 금지)\n"
            "✓ 주제의 의미를 깊이 이해하고 넓게 해석\n"
            "✓ 발언의 표면이 아닌 내용의 본질로 판단\n"
            "✓ 관련 발언이 있으면 반드시 요약 작성\n"
            "✓ JSON 형식만 출력, 다른 텍스트 금지\n\n"
            "응답 JSON 스키마:\n"
            f"{response_schema}"
        )
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        raw_response = response.choices[0].message.content
        parsed = json.loads(raw_response)
        return self._resolve_result(parsed, user_messages, topics)

    def _resolve_result(
        self,
        gpt_output: Dict,
        user_messages: List[Dict],
        topics: List[Dict[str, Optional[str]]],
    ) -> Dict:
        """GPT 응답의 message_id를 실제 발언 텍스트로 변환하고 구조화"""

        # message_id -> 메시지 내용 매핑
        message_map = {msg["id"]: msg for msg in user_messages}
        message_map.update({str(msg["id"]): msg for msg in user_messages})

        resolved_topics = []

        for topic_data in gpt_output.get("topics", []):
            topic_name = topic_data.get("topic", "").strip()
            relevance_score = topic_data.get("relevance_score", 0.0)
            summary = topic_data.get("summary", "").strip()

            # 관련 발언 message_id를 실제 텍스트로 변환
            related_statements = []
            for msg_id in topic_data.get("related_message_ids", []):
                # message_id로 실제 메시지 찾기
                lookup_key = msg_id if msg_id in message_map else str(msg_id)
                msg = message_map.get(lookup_key)

                if msg:
                    related_statements.append(msg["text"])

            # 관련 발언이 없으면 명시
            if not related_statements:
                summary = "이 주제와 관련된 발언이 없습니다."

            resolved_topics.append({
                "topic": topic_name,
                "relevance_score": round(relevance_score, 2),
                "related_statements": related_statements,
                "summary": summary
            })

        # 원래 주제 순서에 맞춰 정렬
        ordered_topics = []
        for original_topic in topics:
            found = False
            original_name = original_topic["name"].lower().strip()

            for resolved in resolved_topics:
                resolved_name = resolved["topic"].lower().strip()

                # 정확히 일치하거나, GPT가 설명을 포함한 경우 처리
                # 예: "메뉴 개발 및 품질 관리" vs "메뉴 개발 및 품질 관리 - 신메뉴 개발..."
                if resolved_name == original_name or resolved_name.startswith(original_name + " -"):
                    # 주제 이름을 원본으로 교체
                    resolved["topic"] = original_topic["name"]
                    ordered_topics.append(resolved)
                    found = True
                    break

            if not found:
                ordered_topics.append({
                    "topic": original_topic["name"],
                    "relevance_score": 0.0,
                    "related_statements": [],
                    "summary": "이 주제와 관련된 발언이 없습니다."
                })

        return {
            "user_id": gpt_output.get("user_id", ""),
            "topics": ordered_topics
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
                    "relevance_score": 0.0,
                    "related_statements": [],
                    "summary": "이 주제와 관련된 발언이 없습니다."
                }
                for topic in topics
            ],
            "generated_at": datetime.now().isoformat(),
        }
