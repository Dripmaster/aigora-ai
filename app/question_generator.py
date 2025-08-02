import random
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()

class QuestionGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.gpt_enabled = True
            self.model = "gpt-4o-mini"
            print(f"QuestionGenerator: OpenAI API 키 설정 완료")
        else:
            self.gpt_enabled = False
            print("QuestionGenerator: OpenAI API 키 설정 실패")

        self.system_prompt = """당신은 CJ 식음 서비스 매니저 교육 프로그램의 전문 토론 사회자입니다.

역할:
- 한국 비즈니스 문화에 맞는 정중하면서도 친근한 화법 사용
- 경어를 사용하되 딱딱하지 않게
- 참여자의 의견을 존중하고 격려하는 태도

말투 예시:
- "~하시는군요", "~해주시면 어떨까요?", "하신 것 같은데요"
- 상대를 높이는 표현 사용("말씀해주신", "의견주신")
- 부드러운 제안형("~하시면 좋을 것 같아요")

주의사항:
- 한국식 존댓말 사용
- 이모지를 적절하게 활용
- 문장 구조는 한국식 문장 구조 사용
- 문장 길이는 너무 짧지도 않고 길지도 않게 적당히
- 질문은 반드시 한 문장으로
"""

        self.templates = {
            "first": [
                "환영합니다! 오늘 토론 주제에 대해 어떻게 생각하시는지 편하게 말씀해주세요",
                "안녕하세요! 토론에 참여해주셔서 감사합니다. 주제에 대한 첫 번째 생각을 들려주세요",
                "반갑습니다! 오늘의 주제와 관련해서 경험하신 일이나 생각이 있으시다면 나눠주세요"
            ],

            "long_idle": [
                "{nickname}님, 아직 계신가요? 토론 주제에 대한 생각을 나눠주세요",
                "다양한 의견이 나오고 있는데요, {nickname}님은 어떻게 생각하시나요?",
                "{nickname}님의 소중한 의견도 듣고 싶어요. 어떤 생각이 드시나요?",
                "잠시 조용하신 {nickname}님, 혹시 궁금한 점이나 의견이 있으시면 말씀해주세요"
            ],

            "only_reaction": [
                "{nickname}님, 공감 표시 감사합니다! 구체적으로 어떤 점이 좋으셨나요?",
                "공감 버튼을 누르신 {nickname}님, 좋은 의견이 있으신가요?",
                "{nickname}님께서 공감해주신 부분에 대해 더 자세한 생각을 들려주시면 어떨까요?",
                "리액션 감사해요, {nickname}님! 그 부분에 대한 개인적인 경험도 있으시다면 나눠주세요"
            ]
        }

    def optimized_prompt(self, user_id: str, nickname: str, idle_time: int, context: Dict) -> str:
        recent_messages = context.get("recent_messages", [])
        recent_chat = ""
        if recent_messages:
            recent_chat = "최근 토론 내용:\n"
            for msg in recent_messages[-5:]:
                recent_chat += f"- {msg.get('user', '참여자')}: {msg.get('text', '')}\n"

        message_count = context.get("message_count", 0)
        reaction_count = context.get("reaction_count", 0)

        user_status = ""
        if message_count == 0 and reaction_count == 0:
            user_status = "첫 참여 대기중"
        elif message_count == 0 and reaction_count > 0:
            user_status = f"공감만 {reaction_count}회 표시"
        else:
            user_status = f"{idle_time}초간 미발언"

        cj_values = """CJ 인재상:
- 정직: 투명하고 진실된 소통
- 열정: 적극적이고 헌신적인 자세
- 창의: 혁신적이고 새로운 접근
- 존중: 고객과 동료를 배려하는 마음
"""

        prompt = f"""[토론 정보]
주제: {context.get('current_topic', 'CJ 인재상 실천')}
상황: {context.get('video_context', '서비스 현장 사례')}

{cj_values}

{recent_chat}

[대상 참여자]
이름: {nickname}
상태: {user_status}

{nickname}님이 토론에 자연스럽게 참여할 수 있도록 한국어로 정중하고 친근한 한 문장으로 작성하세요.
- 구체적이고 답하기 쉬운 질문
- 토론 흐름과 연결된 질문
- 부담이 없는 열린 질문"""

        return prompt

    def generate_gpt_question(self, user_id: str, nickname: str, idle_time: int, context: Dict) -> Optional[str]:
        if not self.gpt_enabled:
            return None

        try:
            prompt = self.optimized_prompt(user_id, nickname, idle_time, context)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=80,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3 
                #임의로 지정 
            )

            question = response.choices[0].message.content.strip()
            print(f"[GPT 생성] {nickname}님께: {question}")
            return question

        except Exception as e:
            print(f"GPT API 오류: {e}")
            return None

    def generate_fallback(self, nickname: str, context: Dict) -> str:
        message_count = context.get("message_count", 0)
        reaction_count = context.get("reaction_count", 0)

        if message_count == 0 and reaction_count == 0:
            templates = self.templates["first"]
        elif message_count == 0 and reaction_count > 0:
            templates = self.templates["only_reaction"]
        else:
            templates = self.templates["long_idle"]

        selected = random.choice(templates)
        question = selected.format(nickname=nickname)
        return question

    def generate(self, user_id: str, nickname: str, idle_time: int, context: Optional[Dict] = None) -> str:
        if context is None:
            context = {}

        if self.gpt_enabled and context.get("use_gpt", True):
            gpt_question = self.generate_gpt_question(user_id, nickname, idle_time, context)
            if gpt_question:
                return gpt_question

        fallback_question = self.generate_fallback(nickname, context)
        print(f"[템플릿 생성] {nickname}님께: {fallback_question}")
        return fallback_question