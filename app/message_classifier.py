import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

class MessageClassifier:
    """
    CJ 식음 서비스 매니저 교육용 메시지 분류기
    토론 참여자의 발언을 CJ 인재상과 성격 특성별로 분류하여 대시보드 분석용 데이터 제공
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.gpt_enabled = True
            self.model = "gpt-4o-mini"
            print(f"MessageClassifier: OpenAI API 키 설정 완료")
        else:
            self.gpt_enabled = False
            print("MessageClassifier: OpenAI API 키 설정 실패 - 룰 기반 분류 사용")

        # CJ 인재상 정의
        self.cj_values = {
            "정직": ["투명한", "솔직한", "진실한", "거짓없는", "정확한", "명확한", "사실", "실제"],
            "열정": ["적극적인", "헌신적인", "노력하는", "열심히", "최선을", "도전", "성장", "발전"],
            "창의": ["혁신적인", "새로운", "독창적인", "아이디어", "개선", "변화", "개혁", "차별화"],
            "존중": ["배려하는", "이해하는", "소통", "협력", "팀워크", "공감", "경청", "도움"]
        }


        # GPT 시스템 프롬프트
        self.system_prompt = """당신은 CJ 식음 서비스 매니저 교육 프로그램의 메시지 분류 전문가입니다.

참여자의 토론 발언을 다음 기준으로 분류해주세요:

**CJ 인재상 (각각 0-100 점수):**
- 정직: 투명하고 진실된 소통을 보여주는 정도
- 열정: 적극적이고 헌신적인 자세를 보여주는 정도  
- 창의: 혁신적이고 새로운 접근을 보여주는 정도
- 존중: 고객과 동료를 배려하는 마음을 보여주는 정도


응답 형식은 반드시 다음 JSON 형태로만 제공하세요:
{
  "cj_values": {
    "정직": 점수,
    "열정": 점수,
    "창의": 점수,
    "존중": 점수
  },
  "primary_trait": "가장_두드러진_CJ_인재상",
  "summary": "한글로_간단한_분류_요약"
}"""

    def classify_with_gpt(self, text: str, user_id: str, context: Optional[Dict] = None) -> Optional[Dict]:
        """GPT를 사용한 메시지 분류"""
        if not self.gpt_enabled:
            return None

        try:
            # 컨텍스트 정보 구성
            context_info = ""
            if context:
                if context.get("previous_messages"):
                    context_info += f"이전 발언들: {context['previous_messages'][-3:]}\n"
                if context.get("discussion_topic"):
                    context_info += f"토론 주제: {context['discussion_topic']}\n"

            user_prompt = f"""{context_info}

분류할 발언: "{text}"
발언자 ID: {user_id}

위 발언을 CJ 인재상 기준으로 분석하여 점수를 매겨주세요."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            print(f"[GPT 분류] {user_id}: {result.get('primary_trait', '분류완료')}")
            return result

        except Exception as e:
            print(f"GPT 분류 오류: {e}")
            return None

    def classify_with_rules(self, text: str, user_id: str) -> Dict:
        """룰 기반 메시지 분류 (GPT 백업용)"""
        text_lower = text.lower()
        
        # CJ 인재상 점수 계산
        cj_scores = {}
        for value, keywords in self.cj_values.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 20
            cj_scores[value] = min(score, 100)


        # 기본 점수 부여 (너무 낮은 점수 방지)
        for value in cj_scores:
            if cj_scores[value] == 0:
                cj_scores[value] = random.randint(10, 30)

        # 주요 특성 찾기
        primary_trait = max(cj_scores, key=cj_scores.get)

        return {
            "cj_values": cj_scores,
            "primary_trait": primary_trait,
            "summary": f"'{primary_trait}' 특성이 두드러진 발언으로 분류됨"
        }

    def classify(self, text: str, user_id: str, context: Optional[Dict] = None) -> Dict:
        """
        메시지 분류 메인 함수
        
        Args:
            text: 분류할 메시지 텍스트
            user_id: 발언자 ID
            context: 추가 컨텍스트 정보 (이전 메시지, 토론 주제 등)
            
        Returns:
            분류 결과 딕셔너리
        """
        # 빈 메시지 처리
        if not text or len(text.strip()) < 2:
            return {
                "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                "primary_trait": "무응답",
                "summary": "메시지가 너무 짧거나 비어있음"
            }

        # GPT 분류 시도
        if self.gpt_enabled and context and context.get("use_gpt", True):
            gpt_result = self.classify_with_gpt(text, user_id, context)
            if gpt_result:
                return gpt_result

        # 룰 기반 분류로 백업
        return self.classify_with_rules(text, user_id)

    def get_user_profile(self, user_id: str, messages: List[Dict]) -> Dict:
        """
        사용자의 전체 발언을 종합하여 종합 프로필 생성
        
        Args:
            user_id: 사용자 ID
            messages: 사용자의 모든 메시지 리스트 [{"text": "...", "timestamp": "..."}, ...]
            
        Returns:
            종합 프로필 딕셔너리
        """
        if not messages:
            return {"error": "분석할 메시지가 없습니다"}

        # 각 메시지 분류
        classifications = []
        for msg in messages:
            result = self.classify(msg["text"], user_id)
            classifications.append(result)

        # 평균 점수 계산
        avg_cj_values = {}
        
        for value in ["정직", "열정", "창의", "존중"]:
            scores = [c["cj_values"][value] for c in classifications if c["cj_values"][value] > 0]
            avg_cj_values[value] = round(sum(scores) / len(scores)) if scores else 0

        # 가장 강한 특성들 찾기
        top_traits = sorted(avg_cj_values.items(), key=lambda x: x[1], reverse=True)[:2]

        return {
            "user_id": user_id,
            "message_count": len(messages),
            "avg_cj_values": avg_cj_values,
            "top_traits": [trait for trait, score in top_traits],
            "overall_summary": f"{user_id}님의 주요 특성: {', '.join([trait for trait, score in top_traits])}"
        }