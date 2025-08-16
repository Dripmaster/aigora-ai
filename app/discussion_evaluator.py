import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from .message_classifier import MessageClassifier

load_dotenv()

class DiscussionEvaluator:
    """
    CJ 식음 서비스 매니저 교육용 토론 총평 생성기
    토론 종료 후 참여자별 종합 평가 및 맞춤형 피드백 제공
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.gpt_enabled = True
            self.model = "gpt-4o-mini"
            print(f"DiscussionEvaluator: OpenAI API 키 설정 완료")
        else:
            self.gpt_enabled = False
            print("DiscussionEvaluator: OpenAI API 키 설정 실패 - 기본 총평 사용")

        # 기존 분류기 활용
        self.classifier = MessageClassifier()

        # GPT 총평 생성 프롬프트
        self.evaluation_prompt = """당신은 CJ 식음 서비스 매니저 교육 프로그램의 전문 평가자입니다.

토론 참여자의 모든 발언을 종합하여 다음 기준으로 상세한 총평을 작성해주세요:

**CJ 인재상 평가 기준:**
- 정직: 투명하고 진실된 소통, 솔직한 의견 표현
- 열정: 적극적이고 헌신적인 자세, 도전 정신
- 창의: 혁신적이고 새로운 접근, 아이디어 제시
- 존중: 고객과 동료를 배려하는 마음, 경청과 공감

**평가 방식:**
1. 각 CJ 인재상별로 0-100점 점수 부여
2. 전체 발언에서 나타난 주요 강점 3개 도출
3. 개선이 필요한 영역 2개 제시
4. 개인화된 발전 방향 제안

**톤 및 스타일:**
- 정중하고 격려적인 존댓말 사용
- 건설적이고 구체적인 피드백
- CJ 가치관과 연결된 조언
- 한국 비즈니스 맥락에 적합한 표현

응답 형식은 반드시 다음 JSON 형태로만 제공하세요:
{
  "overall_score": 전체점수,
  "cj_trait_scores": {
    "정직": 점수,
    "열정": 점수,
    "창의": 점수,
    "존중": 점수
  },
  "participation_summary": "발언패턴과 참여도 요약 (2-3문장)",
  "strengths": ["강점1", "강점2", "강점3"],
  "improvements": ["개선점1", "개선점2"],
  "personalized_feedback": "개인화된 총평 및 발전방향 제안 (4-5문장)",
  "top_messages": ["가장 인상적인 발언 1-2개"]
}"""

    def evaluate_user(self, user_id: str, user_messages: List[Dict], discussion_context: Optional[Dict] = None) -> Dict:
        """
        사용자의 토론 참여 내용을 종합하여 총평 생성
        
        Args:
            user_id: 사용자 ID
            user_messages: 사용자의 모든 발언 리스트 [{"text": "...", "timestamp": "..."}, ...]
            discussion_context: 토론 맥락 정보 (주제, 시간 등)
            
        Returns:
            총평 결과 딕셔너리
        """
        if not user_messages or len(user_messages) == 0:
            return self._create_no_participation_feedback(user_id)

        # GPT 기반 총평 생성 시도
        if self.gpt_enabled:
            gpt_result = self._generate_gpt_evaluation(user_id, user_messages, discussion_context)
            if gpt_result:
                return gpt_result

        # GPT 실패시 기본 총평으로 백업
        return self._generate_fallback_evaluation(user_id, user_messages)

    def _generate_gpt_evaluation(self, user_id: str, user_messages: List[Dict], discussion_context: Optional[Dict] = None) -> Optional[Dict]:
        """GPT를 사용한 종합 총평 생성"""
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

            # 기존 분류기로 각 발언 분석
            classifications = []
            for msg in user_messages:
                result = self.classifier.classify(msg["text"], user_id)
                classifications.append({
                    "text": msg["text"],
                    "primary_trait": result["primary_trait"],
                    "scores": result["cj_values"]
                })

            # 분류 결과 요약
            classification_summary = self._summarize_classifications(classifications)

            user_prompt = f"""{context_info}

평가 대상자: {user_id}
총 발언 수: {len(user_messages)}개

**전체 발언 내역:**
{chr(10).join(messages_text)}

**발언별 CJ 인재상 분석 결과:**
{classification_summary}

위 내용을 종합하여 이 참여자에 대한 상세한 총평을 작성해주세요."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.evaluation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            
            # 결과에 메타데이터 추가
            result["user_id"] = user_id
            result["evaluation_date"] = datetime.now().isoformat()
            result["message_count"] = len(user_messages)
            result["evaluation_method"] = "GPT 기반 종합 평가"

            print(f"[GPT 총평] {user_id}: 종합 점수 {result.get('overall_score', 'N/A')}")
            return result

        except Exception as e:
            print(f"GPT 총평 생성 오류: {e}")
            return None

    def _summarize_classifications(self, classifications: List[Dict]) -> str:
        """발언별 분류 결과를 요약"""
        trait_counts = {"정직": 0, "열정": 0, "창의": 0, "존중": 0}
        high_score_messages = []

        for cls in classifications:
            trait_counts[cls["primary_trait"]] += 1
            
            # 높은 점수 발언 추출
            max_score = max(cls["scores"].values())
            if max_score >= 50:
                high_score_messages.append(f"- {cls['text'][:50]}... ({cls['primary_trait']}: {max_score}점)")

        summary = "주요 인재상 분포: "
        summary += ", ".join([f"{trait} {count}회" for trait, count in trait_counts.items() if count > 0])
        
        if high_score_messages:
            summary += f"\n\n우수 발언 예시:\n" + "\n".join(high_score_messages[:3])
        
        return summary

    def _generate_fallback_evaluation(self, user_id: str, user_messages: List[Dict]) -> Dict:
        """GPT 실패시 기본 총평 생성"""
        # 기존 분류기로 각 발언 분석
        classifications = []
        for msg in user_messages:
            result = self.classifier.classify(msg["text"], user_id)
            classifications.append(result)

        # 평균 점수 계산
        avg_scores = {}
        for trait in ["정직", "열정", "창의", "존중"]:
            scores = [c["cj_values"][trait] for c in classifications if c["cj_values"][trait] > 0]
            avg_scores[trait] = round(sum(scores) / len(scores)) if scores else 0

        overall_score = round(sum(avg_scores.values()) / 4)
        top_trait = max(avg_scores, key=avg_scores.get)

        return {
            "user_id": user_id,
            "overall_score": overall_score,
            "cj_trait_scores": avg_scores,
            "participation_summary": f"총 {len(user_messages)}회 발언하시며 적극적으로 참여해주셨습니다. '{top_trait}' 특성이 가장 두드러지게 나타났습니다.",
            "strengths": [
                f"{top_trait} 인재상이 잘 발현됨",
                "꾸준한 토론 참여",
                "성실한 의견 개진"
            ],
            "improvements": [
                "다양한 CJ 인재상의 균형적 발전",
                "더욱 구체적인 의견 제시"
            ],
            "personalized_feedback": f"{user_id}님은 '{top_trait}' 영역에서 특히 우수한 모습을 보여주셨습니다. 앞으로도 CJ의 핵심 가치를 실천하며 더욱 성장하시길 기대합니다.",
            "top_messages": [msg["text"] for msg in user_messages[:2]],
            "evaluation_date": datetime.now().isoformat(),
            "message_count": len(user_messages),
            "evaluation_method": "룰 기반 기본 평가"
        }

    def _create_no_participation_feedback(self, user_id: str) -> Dict:
        """참여하지 않은 사용자를 위한 피드백"""
        return {
            "user_id": user_id,
            "overall_score": 0,
            "cj_trait_scores": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
            "participation_summary": "이번 토론에 참여하지 않으셨습니다.",
            "strengths": [],
            "improvements": [
                "적극적인 토론 참여",
                "의견 표현 및 소통 활성화"
            ],
            "personalized_feedback": f"{user_id}님, 다음 토론에서는 더욱 적극적으로 참여하여 CJ 인재상을 발휘해보시기 바랍니다. 여러분의 소중한 의견을 기다리고 있습니다.",
            "top_messages": [],
            "evaluation_date": datetime.now().isoformat(),
            "message_count": 0,
            "evaluation_method": "미참여자 기본 안내"
        }

    def evaluate_multiple_users(self, participants_data: Dict[str, List[Dict]], discussion_context: Optional[Dict] = None) -> Dict:
        """
        여러 사용자의 토론 참여 내용을 일괄 평가
        
        Args:
            participants_data: {user_id: [messages...], ...} 형태의 참여자 데이터
            discussion_context: 토론 맥락 정보
            
        Returns:
            전체 참여자 총평 결과
        """
        evaluations = {}
        
        for user_id, messages in participants_data.items():
            print(f"\n{user_id} 총평 생성 중...")
            evaluations[user_id] = self.evaluate_user(user_id, messages, discussion_context)
        
        # 전체 통계 생성
        overall_stats = self._generate_overall_stats(evaluations)
        
        return {
            "discussion_context": discussion_context,
            "individual_evaluations": evaluations,
            "overall_stats": overall_stats,
            "evaluation_date": datetime.now().isoformat()
        }

    def _generate_overall_stats(self, evaluations: Dict) -> Dict:
        """전체 참여자 통계 생성"""
        if not evaluations:
            return {}

        total_participants = len(evaluations)
        active_participants = len([e for e in evaluations.values() if e["message_count"] > 0])
        
        # 평균 점수 계산
        all_scores = [e["overall_score"] for e in evaluations.values() if e["overall_score"] > 0]
        avg_score = round(sum(all_scores) / len(all_scores)) if all_scores else 0
        
        # 상위 참여자 찾기
        sorted_participants = sorted(
            [(uid, eval_data) for uid, eval_data in evaluations.items()],
            key=lambda x: x[1]["overall_score"],
            reverse=True
        )
        
        top_performers = []
        for user_id, eval_data in sorted_participants[:3]:
            if eval_data["overall_score"] > 0:
                top_performers.append({
                    "user_id": user_id,
                    "score": eval_data["overall_score"],
                    "top_trait": max(eval_data["cj_trait_scores"], key=eval_data["cj_trait_scores"].get)
                })

        return {
            "total_participants": total_participants,
            "active_participants": active_participants,
            "average_score": avg_score,
            "top_performers": top_performers,
            "participation_rate": round(active_participants / total_participants * 100, 1) if total_participants > 0 else 0
        }