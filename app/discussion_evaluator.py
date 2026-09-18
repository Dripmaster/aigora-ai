from app.financial_prompts import PERSONAL_RULES, OVERALL_RULES
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from app.financial_profile import AXES, AXIS_PROMPT, FACILITATOR_PROMPT, RUBRIC_PROMPT, normalized_scores

load_dotenv()

class PersonalEvaluator:
    """
    자립준비청년 금융교육 교육용 개인 토론 총평 생성기
    개별 참여자의 토론 성과를 종합 평가하여 맞춤형 피드백 제공
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, timeout=20.0, max_retries=0)
            self.gpt_enabled = True
            self.model = "gpt-4o-mini"
            print(f"PersonalEvaluator: OpenAI API 키 설정 완료")
        else:
            self.gpt_enabled = False
            print("PersonalEvaluator: OpenAI API 키 설정 실패 - 기본 총평 사용")

        # 기존 분류기 활용


        # GPT 개인 총평 생성 프롬프트
        self.evaluation_prompt = FACILITATOR_PROMPT + '\n' + RUBRIC_PROMPT + '\n' + PERSONAL_RULES + '\n제공된 발언 근거로 4축을 0~100 정수로 평가하세요. 세부 기준은 검수 전 초안입니다.\n참여 횟수만으로 역량이나 실제 실행을 단정하지 마세요. 반드시 다음 JSON 형식으로 응답하세요:\n{"overall_score":0,"cj_trait_scores":{"금융이해":0,"위험인식":0,"계획성":0,"실천의지":0},"participation_summary":"발언 근거","strengths":[],"improvements":[],"personalized_feedback":"근거에 따른 피드백","top_messages":[]}'

    def evaluate_user(self, user_id: str, user_messages: List[Dict], discussion_context: Optional[Dict] = None) -> Dict:
        """
        개별 사용자의 토론 참여를 종합 분석하여 개인 맞춤형 총평 생성

        Args:
            user_id: 사용자 ID
            user_messages: 사용자의 모든 발언 리스트 [{"text": "...", "timestamp": "..."}, ...]
            discussion_context: 토론 맥락 정보 (주제, 시간 등)

        Returns:
            개인 맞춤형 총평 결과 딕셔너리
        """
        if not user_messages or len(user_messages) == 0:
            return self._create_no_participation_feedback(user_id)

        # GPT 기반 개인 맞춤 총평 생성 시도
        if self.gpt_enabled:
            gpt_result = self._generate_personal_evaluation(user_id, user_messages, discussion_context)
            if gpt_result:
                return gpt_result

        # GPT 실패시 기본 개인 총평으로 백업
        return self._generate_personal_fallback(user_id, user_messages)

    def _generate_personal_evaluation(self, user_id: str, user_messages: List[Dict], discussion_context: Optional[Dict] = None) -> Optional[Dict]:
        """GPT를 사용한 개인 맞춤형 총평 생성"""
        try:
            # 사용자 발언 데이터 구성
            messages_text = []
            for i, msg in enumerate(user_messages, 1):
                timestamp = msg.get("timestamp", "시간정보없음")
                text = msg.get("text", "")
                messages_text.append(f"{i}. [{timestamp}] {text}")

            # 토론 맥락 정보 구성
            context_info = ""
            if discussion_context:
                if discussion_context.get("topic"):
                    context_info += f"토론 주제: {discussion_context['topic']}\n"
                if discussion_context.get("duration"):
                    context_info += f"토론 시간: {discussion_context['duration']}분\n"
                if discussion_context.get("total_participants"):
                    context_info += f"전체 참여자: {discussion_context['total_participants']}명\n"

            classification_summary = "위 발언 원문과 금융교육 4축 기준을 직접 대조하세요."

            user_prompt = f"""{context_info}

개인 평가 대상: {user_id}님
총 발언 수: {len(user_messages)}개

**개인 발언 전체 내역:**
{chr(10).join(messages_text)}

**개인별 금융교육 발언 4축 발현 분석:**
{classification_summary}

위 발언의 강점과 보완점을 금융교육 4축 기준으로 분석하고, 학생의 목표와 조건에 맞는 구체적인 다음 행동을 제안하세요. 근거 없는 강점은 빈 배열로 남기세요."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.evaluation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            result["cj_trait_scores"] = normalized_scores(result.get("cj_trait_scores"))
            # Preserve the public list-of-strings contract even when the model
            # separates an action and its reason into an object.
            for field in ('strengths', 'improvements'):
                items = result.get(field, [])
                if not isinstance(items, list):
                    items = [items]
                normalized = []
                for item in items:
                    if isinstance(item, dict):
                        item = ' '.join(str(item[key]).strip() for key in
                                        ('strength', 'improvement', 'reason')
                                        if isinstance(item.get(key), str) and item[key].strip())
                    if isinstance(item, str) and item.strip():
                        normalized.append(item.strip())
                result[field] = normalized
            # 결과에 메타데이터 추가
            result["user_id"] = user_id
            result["evaluation_date"] = datetime.now().isoformat()
            result["message_count"] = len(user_messages)
            result["evaluation_method"] = "GPT 기반 개인 맞춤 평가"

            print(f"[GPT 개인총평] {user_id}: 종합 점수 {result.get('overall_score', 'N/A')}")
            return result

        except Exception as e:
            print(f"GPT 개인 총평 생성 오류: {e}")
            return None

    def _summarize_classifications(self, classifications: List[Dict]) -> str:
        """개인별 발언 분류 결과를 요약"""
        trait_counts = {"금융이해": 0, "위험인식": 0, "계획성": 0, "실천의지": 0}
        high_score_messages = []

        for cls in classifications:
            trait_counts[cls["primary_trait"]] += 1

            # 높은 점수 발언 추출
            max_score = max(cls["scores"].values())
            if max_score >= 50:
                high_score_messages.append(f"- {cls['text'][:50]}... ({cls['primary_trait']}: {max_score}점)")

        summary = "개인 발언의 금융 4축 근거: "
        summary += ", ".join([f"{trait} {count}회" for trait, count in trait_counts.items() if count > 0])

        if high_score_messages:
            summary += f"\n\n특징적 발언 예시:\n" + "\n".join(high_score_messages[:3])

        return summary

    def _generate_personal_fallback(self, user_id, user_messages):
        return {
            "user_id": user_id, "overall_score": 0, "cj_trait_scores": dict.fromkeys(AXES, 0),
            "participation_summary": f"기록된 발언은 {len(user_messages)}개입니다. AI 평가를 완료하지 못했습니다.",
            "strengths": [], "improvements": [],
            "personalized_feedback": "AI 평가 결과가 없습니다. 강사 확인 또는 재시도가 필요합니다.",
            "top_messages": [], "evaluation_date": datetime.now().isoformat(),
            "message_count": len(user_messages), "evaluation_method": "평가불가"
        }

    def _create_no_participation_feedback(self, user_id: str) -> Dict:
        """참여하지 않은 사용자를 위한 피드백"""
        return {
            "user_id": user_id,
            "overall_score": 0,
            "cj_trait_scores": {"금융이해": 0, "위험인식": 0, "계획성": 0, "실천의지": 0},
            "participation_summary": "이번 토론에서 평가할 발언 기록이 없습니다.",
            "strengths": [],
            "improvements": [],
            "personalized_feedback": "발언 기록만으로는 평가할 수 없습니다. 준비되면 영상 내용에 대한 의견이나 질문을 남겨주세요.",
            "top_messages": [],
            "evaluation_date": datetime.now().isoformat(),
            "message_count": 0,
            "evaluation_method": "미참여자 개인 맞춤 안내"
        }

    def get_evaluation_summary(self, user_id: str, user_messages: List[Dict], discussion_context: Optional[Dict] = None) -> str:
        """
        개인 총평의 간단한 요약 텍스트 반환 (외부 시스템 연동용)

        Args:
            user_id: 사용자 ID
            user_messages: 사용자 발언 리스트
            discussion_context: 토론 맥락 정보

        Returns:
            개인 총평 요약 텍스트
        """
        evaluation = self.evaluate_user(user_id, user_messages, discussion_context)

        summary_text = f"""
🎯 {user_id}님 개인 토론 총평
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💯 종합 점수: {evaluation['overall_score']}점
📊 금융교육 발언 4축 점수: 금융이해 {evaluation['cj_trait_scores']['금융이해']}점 | 위험인식 {evaluation['cj_trait_scores']['위험인식']}점 | 계획성 {evaluation['cj_trait_scores']['계획성']}점 | 실천의지 {evaluation['cj_trait_scores']['실천의지']}점

📝 참여 요약:
{evaluation['participation_summary']}

✨ 개인 강점:
{chr(10).join([f'• {strength}' for strength in evaluation['strengths']])}

🔄 발전 영역:
{chr(10).join([f'• {improvement}' for improvement in evaluation['improvements']])}

💬 맞춤형 피드백:
{evaluation['personalized_feedback']}

📅 평가일: {evaluation['evaluation_date'][:10]}
🔧 평가방식: {evaluation['evaluation_method']}
        """

        return summary_text.strip()

    def evaluate_discussion_overall(self, all_user_messages: List[Dict], discussion_context: Optional[Dict] = None) -> Dict:
        """
        전체 사용자들의 토론 참여를 종합 분석하여 AI 총평 생성

        Args:
            all_user_messages: 모든 사용자의 발언 리스트 [{"user_id": "...", "text": "...", "timestamp": "..."}, ...]
            discussion_context: 토론 맥락 정보 (주제, 시간 등)

        Returns:
            전체 토론 AI 총평 결과 딕셔너리
        """
        if not all_user_messages or len(all_user_messages) == 0:
            return self._create_no_discussion_feedback()

        # GPT 기반 전체 토론 총평 생성 시도
        if self.gpt_enabled:
            gpt_result = self._generate_discussion_overall_evaluation(all_user_messages, discussion_context)
            if gpt_result:
                return gpt_result

        # GPT 실패시 기본 전체 총평으로 백업
        return self._generate_discussion_overall_fallback(all_user_messages)

    def _generate_discussion_overall_evaluation(self, all_user_messages: List[Dict], discussion_context: Optional[Dict] = None) -> Optional[Dict]:
        """GPT를 사용한 전체 토론 AI 총평 생성"""
        try:
            # 참여자별 발언 분석
            user_participation = {}
            for msg in all_user_messages:
                user_id = (msg.get("user_id") or msg.get("nickname") or msg.get("speaker") or "익명")
                if user_id not in user_participation:
                    user_participation[user_id] = []
                user_participation[user_id].append(msg)

            # 토론 맥락 정보 구성
            context_info = ""
            if discussion_context:
                if discussion_context.get("topic"):
                    context_info += f"토론 주제: {discussion_context['topic']}\n"
                if discussion_context.get("duration"):
                    context_info += f"토론 시간: {discussion_context['duration']}분\n"
                if discussion_context.get("round_number"):
                    context_info += f"토론 회차: {discussion_context['round_number']}차\n"

            # 전체 발언 요약
            total_messages = len(all_user_messages)
            total_users = len(user_participation)

            # 참여자별 발언 수
            participation_summary = []
            for user_id, messages in user_participation.items():
                participation_summary.append(f"{user_id}: {len(messages)}회")

            # 전체 토론 내용 구성 (모든 발언 포함)
            discussion_content = []
            for msg in all_user_messages:
                user_id = (msg.get("user_id") or msg.get("nickname") or msg.get("speaker") or "익명")
                text = msg.get("text", "")
                discussion_content.append(f"- {user_id}: {text}")

            discussion_overall_prompt = FACILITATOR_PROMPT + '\n' + AXIS_PROMPT + '\n' + OVERALL_RULES

            user_prompt = f"""{context_info}

**토론 참여 현황:**
총 참여자: {total_users}명
총 발언 수: {total_messages}개
참여자별 발언: {', '.join(participation_summary)}

**토론 주요 내용:**
{chr(10).join(discussion_content)}

위 기록의 핵심 판단을 분석하고 잘한 점과 보완할 점, 수업 이후 적용할 구체적인 행동을 제시하세요. 모든 참여자가 이해했거나 실천했다고 단정하지 마세요."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": discussion_overall_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            result["cj_trait_scores"] = normalized_scores(result.get("cj_trait_scores"))
            # 결과에 메타데이터 추가
            result["total_participants"] = total_users
            result["total_messages"] = total_messages
            result["evaluation_date"] = datetime.now().isoformat()
            result["evaluation_method"] = "GPT 기반 토론 전체 평가"

            print(f"[GPT 토론총평] 참여자 {total_users}명, 발언 {total_messages}개 분석 완료")
            return result

        except Exception as e:
            print(f"GPT 토론 전체 총평 생성 오류: {e}")
            return None

    def _generate_discussion_overall_fallback(self, all_user_messages: List[Dict]) -> Dict:
        """GPT 실패시 기본 토론 전체 총평 생성"""
        # 참여자별 분석
        user_participation = {}
        total_messages = len(all_user_messages)

        for msg in all_user_messages:
            user_id = (msg.get("user_id") or msg.get("nickname") or msg.get("speaker") or "익명")
            if user_id not in user_participation:
                user_participation[user_id] = 0
            user_participation[user_id] += 1

        total_users = len(user_participation)
        avg_messages_per_user = round(total_messages / total_users) if total_users > 0 else 0

        # 활발한 참여자 파악
        active_users = [user for user, count in user_participation.items() if count >= avg_messages_per_user]

        return {
            "discussion_summary": f"{total_users}명이 {total_messages}개 의견을 나눠주셨습니다. AI 총평을 완료하지 못했습니다. 발언 수 외의 평가 결과는 없습니다.",
            "total_participants": total_users,
            "total_messages": total_messages,
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "평가불가: 발언 수 집계만 제공"
        }

    def _create_no_discussion_feedback(self) -> Dict:
        """토론 참여가 없는 경우 피드백"""
        return {
            "discussion_summary": "이번 토론의 발언 기록이 없어 전체 내용을 요약할 수 없습니다.",
            "total_participants": 0,
            "total_messages": 0,
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "미진행 토론 기본 안내"
        }
