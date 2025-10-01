from typing import Dict, List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import random

load_dotenv()

class ParticipantMonitor:
    """
    참여자 독려 멘트 생성 전담 AI (조건 6)

    핵심 역할:
    - 전체 채팅 내용 모니터링
    - 참여자별 활동 패턴 분석
    - 낙오 위험 감지
    - 참여 독려 멘트 생성
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.gpt_enabled = True
            self.model = "gpt-4o-mini"  # Note: gpt-5-mini 출시 시 변경 가능
            print(f"ParticipantMonitor: OpenAI API 키 설정 완료")
        else:
            self.gpt_enabled = False
            print("ParticipantMonitor: OpenAI API 키 설정 실패 - 템플릿 모드로 작동")

        # 참여자 추적 데이터
        self.participants = {}  # {nickname: ParticipantData}
        self.chat_history = []  # 전체 채팅 기록
        self.intervention_history = {}  # 독려 기록

        # AI 페르소나: 참여 독려 전문가
        self.encouragement_persona = """당신은 **CJ 식음 교육 토론의 참여 독려 전문가**입니다.

**핵심 역할:**
참여가 저조한 사람들에게 부담 없이 토론에 참여할 수 있도록 따뜻하고 친근한 독려 멘트를 생성합니다.

**독려 멘트 생성 원칙:**

1. **친근하고 따뜻한 톤**
   - 부담 주지 않는 부드러운 말투
   - 격려와 환대가 느껴지는 표현
   - 존댓말 사용하되 딱딱하지 않게

2. **참여 수준별 맞춤 독려**
   - 미참여자: "OOO님도 함께 이야기 나눠주시면 좋겠어요 😊"
   - 소극적 참여자: "OOO님 생각도 궁금한데요! 💡"
   - 리액션만 한 사람: "OOO님, 공감해주셔서 감사해요! 👏 구체적인 생각도 나눠주실래요?"

3. **공개 채팅방 형식**
   - 모든 멘트는 전체 채팅방에 공개됨
   - 반드시 닉네임으로 시작 (예: "김매니저님,")
   - 다른 참여자들도 볼 수 있음을 고려

4. **이모지 적극 활용**
   - 친근감을 주는 이모지 1~2개 포함
   - 😊, 💡, ✨, 👏, 💬, 🤔, 😄 등 긍정적 이모지

5. **부담 없는 질문**
   - 강요하지 않는 초대형 질문
   - "~하시면 어떨까요?", "~해주시면 좋겠어요"
   - 열린 질문으로 자유로운 참여 유도

**독려 멘트 유형:**

**1단계 - 부드러운 초대 (미참여자, 1차)**
- "OOO님도 함께 의견 나눠주시면 좋겠어요! 😊"
- "OOO님 생각도 듣고 싶어요! 💡 편하게 말씀해주세요"
- "OOO님, 어떻게 생각하시는지 궁금해요! ✨"

**2단계 - 직접 호명 (미참여자, 2차 이상)**
- "OOO님, 혹시 비슷한 경험 있으셨나요? 🤔"
- "OOO님의 의견이 정말 궁금해요! 😄 나눠주실래요?"
- "OOO님, 다른 분들 의견 중 공감되는 부분 있으셨나요? 💬"

**3단계 - 배려 확인 (지속 미참여)**
- "OOO님, 토론 내용 잘 따라오고 계신가요? 😊"
- "OOO님, 혹시 궁금한 점이나 어려운 부분 있으시면 편하게 말씀해주세요! 🙏"
- "OOO님, 괜찮으신가요? 도움이 필요하시면 언제든 말씀해주세요! 💙"

**금기사항:**
- ❌ 압박하거나 강요하는 톤
- ❌ 부정적이거나 비난하는 표현
- ❌ "왜 참여 안 하세요?" 같은 직접적 지적
- ❌ 참여 부족을 공개적으로 언급
- ❌ 닉네임 없이 멘트 시작

**목표:**
모든 참여자가 안전하고 편안하게 토론에 참여하도록 따뜻한 독려 멘트 제공
"""

        self.system_prompt = self.encouragement_persona

    # ========== 참여자 추적 메서드 ==========

    def update_chat_history(self, nickname: str, text: str):
        """
        채팅 메시지 기록

        Args:
            nickname: 발언자 닉네임
            text: 메시지 내용
        """
        message = {
            "nickname": nickname,
            "text": text
        }
        self.chat_history.append(message)

        # 참여자 정보 업데이트
        if nickname not in self.participants:
            self.participants[nickname] = {
                "message_count": 0,
                "reaction_count": 0,
                "intervention_count": 0
            }

        self.participants[nickname]["message_count"] += 1

    def update_reaction(self, nickname: str):
        """
        리액션(공감 등) 기록

        Args:
            nickname: 참여자 닉네임
        """
        if nickname not in self.participants:
            self.participants[nickname] = {
                "message_count": 0,
                "reaction_count": 0,
                "intervention_count": 0
            }

        self.participants[nickname]["reaction_count"] += 1

    def add_participant(self, nickname: str):
        """
        참여자 등록 (토론방 입장)

        Args:
            nickname: 참여자 닉네임
        """
        if nickname not in self.participants:
            self.participants[nickname] = {
                "message_count": 0,
                "reaction_count": 0,
                "intervention_count": 0
            }
            print(f"[참여자 등록] {nickname}님이 토론에 참여했습니다.")

    # ========== 분석 메서드 ==========

    def get_participant_status(self, nickname: str) -> Dict:
        """
        개별 참여자 상태 분석

        Returns:
            {
                "nickname": str,
                "status": "active|normal|passive|silent",
                "message_count": int,
                "reaction_count": int,
                "engagement_level": int  # 0-3
            }
        """
        if nickname not in self.participants:
            return {
                "nickname": nickname,
                "status": "unknown",
                "message_count": 0,
                "reaction_count": 0,
                "engagement_level": 0
            }

        data = self.participants[nickname]
        message_count = data["message_count"]
        reaction_count = data["reaction_count"]

        # 상태 분류
        status = "normal"
        engagement_level = 1

        if message_count >= 5:
            status = "active"  # 활발
            engagement_level = 3
        elif message_count >= 2:
            status = "normal"  # 보통
            engagement_level = 2
        elif message_count == 1 or reaction_count > 0:
            status = "passive"  # 소극적
            engagement_level = 1
        else:
            status = "silent"  # 침묵
            engagement_level = 0

        return {
            "nickname": nickname,
            "status": status,
            "message_count": message_count,
            "reaction_count": reaction_count,
            "engagement_level": engagement_level
        }

    def get_all_participants_status(self) -> List[Dict]:
        """전체 참여자 상태 리스트"""
        statuses = []
        for nickname in self.participants.keys():
            status = self.get_participant_status(nickname)
            statuses.append(status)

        # 참여도 낮은 순으로 정렬
        statuses.sort(key=lambda x: x["engagement_level"])
        return statuses

    def get_silent_participants(self) -> List[Dict]:
        """침묵/소극적 참여자 리스트"""
        all_status = self.get_all_participants_status()
        return [s for s in all_status if s["engagement_level"] <= 1]

    def get_summary_stats(self) -> Dict:
        """전체 토론 참여 통계"""
        all_status = self.get_all_participants_status()

        stats = {
            "total": len(all_status),
            "active": len([s for s in all_status if s["status"] == "active"]),
            "normal": len([s for s in all_status if s["status"] == "normal"]),
            "passive": len([s for s in all_status if s["status"] == "passive"]),
            "silent": len([s for s in all_status if s["status"] == "silent"]),
            "avg_messages": sum(s["message_count"] for s in all_status) / len(all_status) if all_status else 0
        }

        return stats

    # ========== 독려 멘트 생성 메서드 ==========

    def should_encourage(self, nickname: str) -> Dict:
        """
        독려 필요 여부 판단

        Returns:
            {
                "should_encourage": bool,
                "encouragement_level": int,  # 1-3
                "reason": str
            }
        """
        status = self.get_participant_status(nickname)
        intervention_count = self.participants[nickname].get("intervention_count", 0)

        should_encourage = False
        encouragement_level = 1
        reason = ""

        if status["status"] == "silent":
            should_encourage = True
            encouragement_level = min(intervention_count + 1, 3)
            reason = f"침묵 중, {intervention_count}회 독려 이력"

        elif status["status"] == "passive":
            should_encourage = True
            encouragement_level = min(intervention_count + 1, 2)
            reason = f"소극적 참여, {intervention_count}회 독려 이력"

        return {
            "should_encourage": should_encourage,
            "encouragement_level": encouragement_level,
            "reason": reason,
            "status": status
        }

    def generate_encouragement_message(self, nickname: str, chat_history: List[Dict],
                                      encouragement_level: int = 1) -> str:
        """
        독려 멘트 생성 (GPT 또는 템플릿)

        Args:
            nickname: 대상 참여자
            chat_history: 전체 채팅 내역
            encouragement_level: 독려 강도 (1=부드러운 초대, 2=직접 호명, 3=배려 확인)

        Returns:
            독려 멘트 문자열
        """
        # GPT 생성 시도
        if self.gpt_enabled:
            try:
                return self._generate_gpt_encouragement(nickname, chat_history, encouragement_level)
            except Exception as e:
                print(f"GPT API 오류: {e}, 템플릿 모드로 전환")

        # 템플릿 폴백
        return self._generate_template_encouragement(nickname, encouragement_level)

    def _generate_gpt_encouragement(self, nickname: str, chat_history: List[Dict],
                                   encouragement_level: int) -> str:
        """GPT 기반 독려 멘트 생성"""
        # 최근 채팅 요약
        recent_chat = ""
        if chat_history:
            recent_count = min(10, len(chat_history))
            recent_chat = f"**최근 토론 흐름 (최근 {recent_count}개 메시지):**\n"
            for msg in chat_history[-recent_count:]:
                recent_chat += f"- {msg.get('nickname', '참여자')}: {msg.get('text', '')}\n"

        level_guide = {
            1: "1단계 - 부드러운 초대: 부담 없이 참여를 유도하는 친근한 멘트",
            2: "2단계 - 직접 호명: 구체적으로 의견을 물어보는 멘트",
            3: "3단계 - 배려 확인: 토론 이해도나 어려움을 확인하는 멘트"
        }

        prompt = f"""[토론 상황]
{recent_chat}

[대상 참여자]
닉네임: {nickname}
독려 단계: {level_guide.get(encouragement_level, level_guide[1])}

[독려 멘트 생성 요청]
{nickname}님이 부담 없이 토론에 참여할 수 있도록 독려 멘트를 만들어주세요.

요구사항:
1. **닉네임 필수**: 반드시 "{nickname}님," 또는 "{nickname}님" 으로 시작
2. **친근하고 따뜻한 톤**: 격려와 환대가 느껴지는 말투
3. **이모지 포함**: 긍정적 이모지 1~2개
4. **부담 없는 표현**: 강요하지 않고 초대하는 느낌
5. **한 문장**: 간결하고 명확하게

**반드시 한국어로, {nickname}님으로 시작하는 한 문장의 독려 멘트만 생성하세요.**"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.4,
            presence_penalty=0.4
        )

        message = response.choices[0].message.content.strip()
        print(f"[GPT 독려 멘트] {nickname}님께: {message}")
        return message

    def _generate_template_encouragement(self, nickname: str, encouragement_level: int) -> str:
        """템플릿 기반 독려 멘트 생성"""
        templates = {
            1: [  # 부드러운 초대
                f"{nickname}님도 함께 의견 나눠주시면 좋겠어요! 😊",
                f"{nickname}님 생각도 듣고 싶어요! 💡 편하게 말씀해주세요",
                f"{nickname}님, 어떻게 생각하시는지 궁금해요! ✨"
            ],
            2: [  # 직접 호명
                f"{nickname}님, 혹시 비슷한 경험 있으셨나요? 🤔",
                f"{nickname}님의 의견이 정말 궁금해요! 😄 나눠주실래요?",
                f"{nickname}님, 다른 분들 의견 중 공감되는 부분 있으셨나요? 💬"
            ],
            3: [  # 배려 확인
                f"{nickname}님, 토론 내용 잘 따라오고 계신가요? 😊",
                f"{nickname}님, 혹시 궁금한 점이나 어려운 부분 있으시면 편하게 말씀해주세요! 🙏",
                f"{nickname}님, 괜찮으신가요? 도움이 필요하시면 언제든 말씀해주세요! 💙"
            ]
        }

        messages = templates.get(encouragement_level, templates[1])
        message = random.choice(messages)
        print(f"[템플릿 독려 멘트] {nickname}님께: {message}")
        return message

    def record_encouragement(self, nickname: str):
        """독려 기록"""
        if nickname not in self.intervention_history:
            self.intervention_history[nickname] = []

        self.intervention_history[nickname].append({
            "count": len(self.intervention_history[nickname]) + 1
        })

        self.participants[nickname]["intervention_count"] += 1
        print(f"[독려 기록] {nickname}님께 {self.participants[nickname]['intervention_count']}차 독려 수행")

    # ========== 메인 인터페이스 ==========

    def check_and_encourage(self, chat_history: Optional[List[Dict]] = None) -> List[Dict]:
        """
        전체 참여자 체크 후 독려 대상 및 멘트 반환

        Args:
            chat_history: 전체 채팅 내역 (선택)

        Returns:
            [
                {
                    "nickname": str,
                    "message": str,
                    "encouragement_level": int,
                    "reason": str
                },
                ...
            ]
        """
        if chat_history is None:
            chat_history = self.chat_history

        encouragements = []

        for nickname in self.participants.keys():
            decision = self.should_encourage(nickname)

            if decision["should_encourage"]:
                message = self.generate_encouragement_message(
                    nickname,
                    chat_history,
                    decision["encouragement_level"]
                )
                encouragements.append({
                    "nickname": nickname,
                    "message": message,
                    "encouragement_level": decision["encouragement_level"],
                    "reason": decision["reason"]
                })

                # 독려 기록
                self.record_encouragement(nickname)

        # 참여도 낮은 순으로 정렬
        encouragements.sort(key=lambda x: -x["encouragement_level"])
        return encouragements
