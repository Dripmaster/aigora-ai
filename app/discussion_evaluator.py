import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from .message_classifier import MessageClassifier

load_dotenv()

class PersonalEvaluator:
    """
    CJ 식음 서비스 매니저 교육용 개인 토론 총평 생성기
    개별 참여자의 토론 성과를 종합 평가하여 맞춤형 피드백 제공
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.gpt_enabled = True
            self.model = "gpt-4o-mini"
            print(f"PersonalEvaluator: OpenAI API 키 설정 완료")
        else:
            self.gpt_enabled = False
            print("PersonalEvaluator: OpenAI API 키 설정 실패 - 기본 총평 사용")

        # 기존 분류기 활용
        self.classifier = MessageClassifier()

        # GPT 개인 총평 생성 프롬프트
        self.evaluation_prompt = """당신은 CJ 식음 서비스 매니저 교육 프로그램의 전문 평가자입니다.

개별 참여자의 토론 발언을 종합 분석하여 개인 맞춤형 총평을 작성해주세요:

**CJ 인재상 개인 평가 기준:**
- 정직: 솔직하고 투명한 의견 표현, 진실된 소통 태도
- 열정: 적극적 참여 의지, 업무에 대한 헌신과 도전 정신
- 창의: 새로운 아이디어 제시, 문제 해결의 혁신적 접근
- 존중: 타인 배려, 경청과 공감을 통한 협력적 자세

**개인 평가 중점 사항:**
1. 개인의 고유한 강점과 특성 파악
2. 토론 참여 패턴과 소통 스타일 분석
3. CJ 인재상 발현 정도를 세밀하게 평가 (0-100점)
4. 개인별 성장 가능성과 발전 방향 제시
5. 실무에 적용 가능한 구체적 조언

**피드백 스타일:**
- 개인의 특성을 인정하고 격려하는 톤
- 강점을 부각시키면서 성장 영역 안내
- CJ 가치와 연결된 실무 적용 가이드
- 한국적 비즈니스 예의를 갖춘 정중한 표현

응답 형식은 반드시 다음 JSON 형태로만 제공하세요:
{
  "overall_score": 전체점수,
  "cj_trait_scores": {
    "정직": 점수,
    "열정": 점수,
    "창의": 점수,
    "존중": 점수
  },
  "participation_summary": "개인의 토론 참여 스타일과 특징 (3-4문장)",
  "strengths": ["개인 고유 강점1", "개인 고유 강점2", "개인 고유 강점3"],
  "improvements": ["개인 맞춤 개선점1", "개인 맞춤 개선점2"],
  "personalized_feedback": "개인별 맞춤 총평과 성장 방향 (5-6문장)",
  "top_messages": ["가장 인상적이고 특징적인 발언 1-2개"]
}"""

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

개인 평가 대상: {user_id}님
총 발언 수: {len(user_messages)}개

**개인 발언 전체 내역:**
{chr(10).join(messages_text)}

**개인별 CJ 인재상 발현 분석:**
{classification_summary}

위 내용을 토대로 {user_id}님만의 고유한 특성과 강점을 파악하여 개인 맞춤형 총평을 작성해주세요.
특히 이 분의 토론 스타일, 소통 방식, CJ 인재상 발현 패턴을 중심으로 개인화된 피드백을 제공해주세요."""

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
            result["evaluation_method"] = "GPT 기반 개인 맞춤 평가"

            print(f"[GPT 개인총평] {user_id}: 종합 점수 {result.get('overall_score', 'N/A')}")
            return result

        except Exception as e:
            print(f"GPT 개인 총평 생성 오류: {e}")
            return None

    def _summarize_classifications(self, classifications: List[Dict]) -> str:
        """개인별 발언 분류 결과를 요약"""
        trait_counts = {"정직": 0, "열정": 0, "창의": 0, "존중": 0}
        high_score_messages = []

        for cls in classifications:
            trait_counts[cls["primary_trait"]] += 1
            
            # 높은 점수 발언 추출
            max_score = max(cls["scores"].values())
            if max_score >= 50:
                high_score_messages.append(f"- {cls['text'][:50]}... ({cls['primary_trait']}: {max_score}점)")

        summary = "개인 인재상 발현 패턴: "
        summary += ", ".join([f"{trait} {count}회" for trait, count in trait_counts.items() if count > 0])
        
        if high_score_messages:
            summary += f"\n\n특징적 발언 예시:\n" + "\n".join(high_score_messages[:3])
        
        return summary

    def _generate_personal_fallback(self, user_id: str, user_messages: List[Dict]) -> Dict:
        """GPT 실패시 기본 개인 총평 생성"""
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
            "participation_summary": f"{user_id}님은 총 {len(user_messages)}회의 발언을 통해 적극적으로 토론에 참여해주셨습니다. 특히 '{top_trait}' 인재상이 두드러지게 발현되어 개인만의 특징적인 소통 스타일을 보여주셨습니다.",
            "strengths": [
                f"개인 고유의 {top_trait} 특성 발현",
                f"{user_id}님만의 독특한 관점과 접근",
                "일관된 토론 참여 의지"
            ],
            "improvements": [
                f"{user_id}님의 강점인 {top_trait}을 더욱 발전시키기",
                "다른 CJ 인재상과의 균형적 통합"
            ],
            "personalized_feedback": f"{user_id}님은 '{top_trait}' 영역에서 개인만의 독특한 강점을 보여주셨습니다. 이는 {user_id}님의 고유한 특성으로, 앞으로 이 강점을 더욱 발전시키면서 다른 인재상들과 조화롭게 통합해나가시면 CJ의 핵심 인재로 더욱 성장하실 것으로 기대됩니다.",
            "top_messages": [msg["text"] for msg in user_messages[:2]],
            "evaluation_date": datetime.now().isoformat(),
            "message_count": len(user_messages),
            "evaluation_method": "룰 기반 개인 맞춤 평가"
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
            "personalized_feedback": f"{user_id}님, 이번에는 토론 참여 기회를 놓치셨지만, 다음 토론에서는 {user_id}님만의 고유한 관점과 CJ 인재상을 마음껏 발휘해보시기 바랍니다. {user_id}님의 개성 있는 의견과 참여를 기대하고 있습니다.",
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
📊 CJ 인재상 점수: 정직 {evaluation['cj_trait_scores']['정직']}점 | 열정 {evaluation['cj_trait_scores']['열정']}점 | 창의 {evaluation['cj_trait_scores']['창의']}점 | 존중 {evaluation['cj_trait_scores']['존중']}점

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
                user_id = msg.get("user_id", "익명")
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

            # 전체 토론 내용 구성 (최근 20개 발언만)
            recent_messages = all_user_messages[-20:] if len(all_user_messages) > 20 else all_user_messages
            discussion_content = []
            for msg in recent_messages:
                user_id = msg.get("user_id", "익명")
                text = msg.get("text", "")
                discussion_content.append(f"- {user_id}: {text}")

            discussion_overall_prompt = """당신은 CJ 식음 서비스 매니저 교육 프로그램의 전문 토론 평가자입니다.

전체 토론 참여자들의 발언을 종합 분석하여 토론 전체에 대한 AI 총평을 작성해주세요:

**토론 전체 평가 기준:**
- 참여도: 전체 참여율과 적극성 분석
- 토론 품질: 발언의 깊이와 CJ 인재상 발현도
- 상호작용: 참여자 간 소통과 협력 정도
- 균형도: 다양한 관점과 의견 표현 정도

**AI 총평 중점 사항:**
1. 전체적인 토론 분위기와 참여 양상
2. 주요 인사이트와 우수 발언들
3. CJ 인재상 발현 패턴 분석
4. 토론의 교육적 성과와 개선점

응답 형식은 반드시 다음 JSON 형태로만 제공하세요:
{
  "overall_quality_score": 전체품질점수,
  "participation_rate": 참여율점수,
  "discussion_summary": "토론 전체 요약 및 특징 (4-5문장)",
  "key_insights": ["주요 인사이트1", "주요 인사이트2", "주요 인사이트3"],
  "top_contributions": ["우수 발언이나 기여 1-2개"],
  "cj_values_reflection": "CJ 인재상 발현 정도 분석 (3-4문장)",
  "recommendations": ["토론 개선점1", "토론 개선점2"]
}"""

            user_prompt = f"""{context_info}

**토론 참여 현황:**
총 참여자: {total_users}명
총 발언 수: {total_messages}개
참여자별 발언: {', '.join(participation_summary)}

**토론 주요 내용:**
{chr(10).join(discussion_content)}

위 토론 내용을 종합하여 전체 토론에 대한 AI 총평을 작성해주세요.
특히 참여자들의 CJ 인재상 발현과 토론의 교육적 가치를 중심으로 평가해주세요."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": discussion_overall_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            
            # 결과에 메타데이터 추가
            result["total_participants"] = total_users
            result["total_messages"] = total_messages
            result["evaluation_date"] = datetime.now().isoformat()
            result["evaluation_method"] = "GPT 기반 토론 전체 평가"

            print(f"[GPT 토론총평] 참여자 {total_users}명, 발언 {total_messages}개: 품질점수 {result.get('overall_quality_score', 'N/A')}")
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
            user_id = msg.get("user_id", "익명")
            if user_id not in user_participation:
                user_participation[user_id] = 0
            user_participation[user_id] += 1

        total_users = len(user_participation)
        avg_messages_per_user = round(total_messages / total_users) if total_users > 0 else 0
        
        # 활발한 참여자 파악
        active_users = [user for user, count in user_participation.items() if count >= avg_messages_per_user]
        
        return {
            "overall_quality_score": min(85, 60 + (total_messages * 2)),  # 발언 수에 따른 기본 점수
            "participation_rate": min(90, 50 + (total_users * 10)),  # 참여자 수에 따른 점수
            "discussion_summary": f"총 {total_users}명이 참여하여 {total_messages}개의 발언으로 활발한 토론이 진행되었습니다. 참여자 평균 {avg_messages_per_user}회 발언으로 적극적인 참여를 보여주었으며, 다양한 관점에서 CJ 인재상에 대한 깊이 있는 논의가 이루어졌습니다.",
            "key_insights": [
                f"{len(active_users)}명의 참여자가 특히 활발한 기여를 보임",
                f"전체적으로 균형잡힌 {total_messages}개 발언으로 구성",
                "다양한 CJ 인재상 관점에서 접근한 토론"
            ],
            "top_contributions": [msg.get("text", "")[:100] + "..." for msg in all_user_messages[:2]],
            "cj_values_reflection": f"참여자들이 CJ의 핵심 가치인 정직, 열정, 창의, 존중을 바탕으로 한 발언들을 통해 실무 현장에서의 적용 방안에 대해 진지하게 고민하는 모습을 보여주었습니다. 전체적으로 CJ 인재상에 대한 이해도가 높은 토론이었습니다.",
            "recommendations": [
                "더 많은 참여자들의 적극적인 발언 유도 필요",
                "구체적인 실무 사례를 통한 심화 토론 권장"
            ],
            "total_participants": total_users,
            "total_messages": total_messages,
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "룰 기반 토론 전체 평가"
        }

    def _create_no_discussion_feedback(self) -> Dict:
        """토론 참여가 없는 경우 피드백"""
        return {
            "overall_quality_score": 0,
            "participation_rate": 0,
            "discussion_summary": "이번 토론에는 참여자나 발언이 없었습니다.",
            "key_insights": ["토론 미진행"],
            "top_contributions": [],
            "cj_values_reflection": "토론이 진행되지 않아 CJ 인재상 발현을 평가할 수 없습니다.",
            "recommendations": [
                "참여자 유도를 위한 적극적인 안내 필요",
                "토론 주제나 방식 재검토 권장"
            ],
            "total_participants": 0,
            "total_messages": 0,
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "미진행 토론 기본 안내"
        }