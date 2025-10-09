import json
import os
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class MessageClassifierGPT:
    """
    CJ 인재상 기반 GPT 메시지 분류기
    OpenAI GPT-4o-mini를 활용하여 토론 참여자의 발언을 CJ 인재상으로 분류
    """

    def __init__(self):
        """GPT 분류기 초기화"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        
        # CJ 인재상 정의
        self.cj_values = {
            "정직": "투명하고 진실된 소통을 보여주는 정도",
            "열정": "적극적이고 헌신적인 자세를 보여주는 정도",
            "창의": "혁신적이고 새로운 접근을 보여주는 정도",
            "존중": "고객과 동료를 배려하는 마음을 보여주는 정도"
        }
        
        print(f"MessageClassifierGPT: OpenAI API 키 설정 완료 (모델: {self.model})")

    def _create_system_prompt(self) -> str:
        """GPT 시스템 프롬프트 생성"""
        return """당신은 CJ 식음 서비스 매니저 교육 프로그램의 메시지 분류 전문가입니다.

참여자의 토론 발언을 다음 CJ 인재상 기준으로 분석하여 점수를 매겨주세요:

**CJ 인재상 정의:**
- 정직(Honesty): 투명하고 진실된 소통을 보여주는 정도 (솔직한 표현, 사실 기반 발언, 투명한 의사소통)
- 열정(Passion): 적극적이고 헌신적인 자세를 보여주는 정도 (적극적 참여, 도전 의식, 헌신적 태도)
- 창의(Creativity): 혁신적이고 새로운 접근을 보여주는 정도 (새로운 아이디어, 혁신적 사고, 독창적 해결책)
- 존중(Respect): 고객과 동료를 배려하는 마음을 보여주는 정도 (배려심, 공감 능력, 협력적 자세)

**채점 기준:**
- 각 인재상마다 0-100점으로 채점
- 발언에서 명확하게 드러나는 특성에만 점수 부여
- 여러 인재상이 동시에 나타날 수 있음
- 한국어 표현의 뉘앙스와 문맥을 정확히 파악

**응답 형식:**
반드시 다음 JSON 형태로만 응답하세요:
{
  "cj_values": {
    "정직": 점수(0-100),
    "열정": 점수(0-100),
    "창의": 점수(0-100),
    "존중": 점수(0-100)
  },
  "primary_trait": "가장_두드러진_CJ_인재상",
  "summary": "한글로_간단한_분류_요약"
}"""

    def _create_user_prompt(self, text: str, user_id: str, context: Optional[Dict] = None) -> str:
        """사용자 프롬프트 생성"""
        prompt_parts = []
        
        # 컨텍스트 정보 추가
        if context:
            if context.get("discussion_topic"):
                prompt_parts.append(f"토론 주제: {context['discussion_topic']}")
            
            if context.get("previous_messages"):
                prev_msgs = context['previous_messages'][-3:]  # 최근 3개 메시지
                prompt_parts.append(f"이전 발언들: {prev_msgs}")
            
            if context.get("user_history"):
                prompt_parts.append(f"발언자 이력: {context['user_history']}")
        
        # 메인 프롬프트
        prompt_parts.extend([
            f"분류할 발언: \"{text}\"",
            f"발언자 ID: {user_id}",
            "",
            "위 발언을 CJ 인재상 기준으로 분석하여 각 특성별 점수를 매겨주세요.",
            "발언에서 직접적으로 드러나는 특성만 점수를 부여하고, 추측하지 마세요."
        ])
        
        return "\n".join(prompt_parts)

    def classify(self, text: str, user_id: str, context: Optional[Dict] = None) -> Dict:
        """
        GPT를 사용하여 메시지를 CJ 인재상으로 분류
        
        Args:
            text: 분류할 메시지 텍스트
            user_id: 발언자 ID
            context: 추가 컨텍스트 정보
                - discussion_topic: 토론 주제
                - previous_messages: 이전 메시지들
                - user_history: 사용자 발언 이력
                
        Returns:
            Dict: 분류 결과
                - cj_values: 각 CJ 인재상별 점수 (0-100)
                - primary_trait: 가장 두드러진 특성
                - summary: 분류 요약
        """
        # 입력 검증
        if not text or len(text.strip()) < 2:
            return {
                "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                "primary_trait": "무응답",
                "summary": "메시지가 너무 짧거나 비어있음"
            }
        
        try:
            # GPT API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": self._create_user_prompt(text, user_id, context)}
                ],
                temperature=0.3,
                max_tokens=400,
                response_format={"type": "json_object"}
            )
            
            # JSON 파싱
            result = json.loads(response.choices[0].message.content)
            
            # 결과 검증 및 정규화
            cj_values = result.get("cj_values", {})
            
            # 점수 범위 검증 및 조정
            for trait in ["정직", "열정", "창의", "존중"]:
                if trait not in cj_values:
                    cj_values[trait] = 0
                else:
                    # 0-100 범위로 제한
                    cj_values[trait] = max(0, min(100, int(cj_values[trait])))
            
            # 주요 특성 재계산
            primary_trait = max(cj_values, key=cj_values.get) if max(cj_values.values()) > 0 else "무응답"
            
            # 최종 결과 구성
            final_result = {
                "cj_values": cj_values,
                "primary_trait": primary_trait,
                "summary": result.get("summary", f"'{primary_trait}' 특성이 두드러진 발언으로 분류됨")
            }
            
            print(f"[GPT 분류] {user_id}: {primary_trait} (점수: {cj_values})")
            return final_result
            
        except json.JSONDecodeError as e:
            print(f"GPT 응답 JSON 파싱 오류: {e}")
            return self._create_fallback_result(text, user_id)
            
        except Exception as e:
            print(f"GPT 분류 오류: {e}")
            return self._create_fallback_result(text, user_id)

    def _create_fallback_result(self, text: str, user_id: str) -> Dict:
        """GPT 오류 시 기본 결과 생성"""
        # 간단한 키워드 기반 분류
        text_lower = text.lower()
        
        basic_keywords = {
            "정직": ["솔직히", "사실", "정말", "진짜", "실제", "현실"],
            "열정": ["열심히", "최선", "노력", "도전", "좋", "대단"],
            "창의": ["새로운", "아이디어", "다른", "혁신", "색다른"],
            "존중": ["배려", "공감", "이해", "소통", "함께", "고객"]
        }
        
        scores = {}
        for trait, keywords in basic_keywords.items():
            score = sum(15 for keyword in keywords if keyword in text_lower)
            scores[trait] = min(score, 100)
        
        primary_trait = max(scores, key=scores.get) if max(scores.values()) > 0 else "무응답"
        
        return {
            "cj_values": scores,
            "primary_trait": primary_trait,
            "summary": f"GPT 분류 실패로 기본 키워드 분류 적용: {primary_trait}"
        }

    def batch_classify(self, messages: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        """
        여러 메시지를 일괄 분류
        
        Args:
            messages: 메시지 리스트 [{"text": "...", "user_id": "..."}, ...]
            context: 공통 컨텍스트 정보
            
        Returns:
            List[Dict]: 각 메시지의 분류 결과 리스트
        """
        results = []
        
        for msg in messages:
            if not isinstance(msg, dict) or "text" not in msg or "user_id" not in msg:
                results.append({
                    "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                    "primary_trait": "오류",
                    "summary": "잘못된 메시지 형식"
                })
                continue
            
            result = self.classify(msg["text"], msg["user_id"], context)
            results.append(result)
        
        return results

    def analyze_user_profile(self, user_id: str, messages: List[str], context: Optional[Dict] = None) -> Dict:
        """
        사용자의 전체 발언을 분석하여 종합 프로필 생성
        
        Args:
            user_id: 사용자 ID
            messages: 사용자의 모든 메시지 텍스트 리스트
            context: 추가 컨텍스트 정보
            
        Returns:
            Dict: 종합 프로필 결과
        """
        if not messages:
            return {"error": "분석할 메시지가 없습니다"}
        
        # 각 메시지 분류
        classifications = []
        for text in messages:
            result = self.classify(text, user_id, context)
            classifications.append(result)
        
        # 평균 점수 계산
        avg_cj_values = {}
        for trait in ["정직", "열정", "창의", "존중"]:
            scores = [c["cj_values"][trait] for c in classifications if c["cj_values"][trait] > 0]
            avg_cj_values[trait] = round(sum(scores) / len(scores)) if scores else 0
        
        # 상위 특성 식별
        top_traits = sorted(avg_cj_values.items(), key=lambda x: x[1], reverse=True)[:2]
        
        # 특성별 발언 수 집계
        trait_counts = {}
        for trait in ["정직", "열정", "창의", "존중"]:
            trait_counts[trait] = sum(1 for c in classifications if c["primary_trait"] == trait)
        
        return {
            "user_id": user_id,
            "message_count": len(messages),
            "avg_cj_values": avg_cj_values,
            "top_traits": [trait for trait, score in top_traits if score > 0],
            "trait_distribution": trait_counts,
            "strongest_trait": top_traits[0][0] if top_traits[0][1] > 0 else "무응답",
            "profile_summary": f"{user_id}님은 '{top_traits[0][0]}' 특성이 가장 강하며, 전체 {len(messages)}개 발언 중 평균 {round(sum(avg_cj_values.values())/4)}점의 CJ 인재상을 보여줌"
        }