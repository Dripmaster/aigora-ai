from app.financial_prompts import ENCOURAGEMENT_RULES
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
            self.client = OpenAI(api_key=self.api_key, timeout=20.0, max_retries=0)
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
        self.recent_messages = []  # 최근 생성된 독려 멘트 (중복 방지용)

        # AI 페르소나: 참여 독려 전문가
        self.encouragement_persona = '''당신은 금융교육 토론의 참여 독려 도우미입니다.
당신의 역할은 학생이 부담 없이 토론에 참여하도록 돕는 것입니다.
따뜻하고 친근한 존댓말로 환대하고, 딱딱한 공지 대신 자연스럽게 대화에 초대하세요.
따뜻함은 학생의 말을 들어주는 표현으로 드러내고, 매번 '멋진 계획', '좋은 선택'이라고 평가하지 마세요.
최근 학생 발언에서 실제 나온 의견이나 궁금한 점을 연결해 토론을 듣고 있음을 보여주세요.
미참여자에게는 부드러운 초대, 의견을 낸 학생에게는 그 의견에 연결한 질문,
리액션만 한 학생에게는 공감한 부분을 말할 수 있는 초대를 하되, 제공된 기록에 있을 때만 언급하세요.
참여 부족을 지적하거나 다른 학생과 비교하지 마세요.
학생이 준비되었을 때 참여할 수 있도록 여유를 주고, 가벼운 이모지도 자연스럽게 활용하세요.
영상이나 교재 원문은 제공되지 않으므로, 학생의 실제 발언만 맥락으로 삼으세요.
입력에 없는 영상 속 행동·계획·사건을 만들어내지 마세요.
''' + '\n' + ENCOURAGEMENT_RULES

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
            status = "passive"  # 발언 기록이 적음
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
        """최근 발언 기록이 적은 참여자 리스트"""
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
            reason = f"발언 기록 적음, {intervention_count}회 안내 이력"

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
                print(f"참여 안내 생성 실패: {type(e).__name__}")

        # 템플릿 폴백
        return self._generate_template_encouragement(nickname, encouragement_level)

    def _generate_gpt_encouragement(self, nickname: str, chat_history: List[Dict],
                                   encouragement_level: int) -> str:
        """GPT 기반 독려 멘트 생성 (다양성 강화)"""
        # 최근 채팅 요약
        recent_chat = ""
        if chat_history:
            recent_count = min(10, len(chat_history))
            recent_chat = f"**최근 토론 흐름 (최근 {recent_count}개 메시지):**\n"
            for msg in chat_history[-recent_count:]:
                recent_chat += f"- {msg.get('nickname', '참여자')}: {msg.get('text', '')}\n"

        level_guide = {
            1: "1단계 - 부드러운 초대: 부담 없이 참여를 유도하는 친근한 멘트",
            2: "2단계 - 구체적인 초대: 현재 토론이나 영상 속 선택에 연결해 의견을 물어보는 멘트",
            3: "3단계 - 배려 확인: 궁금한 점이나 도움이 필요한 부분을 편하게 말할 수 있는 멘트"
        }

        # 최근 생성된 멘트 히스토리 추가 (중복 방지)
        recent_messages_text = ""
        if self.recent_messages:
            recent_messages_text = "\n**최근 생성된 독려 멘트 (중복 방지):**\n"
            for msg in self.recent_messages[-5:]:  # 최근 5개만
                recent_messages_text += f"- {msg}\n"
            recent_messages_text += "\n최근 안내와 의미가 같은 요청을 반복하지 마세요.\n"

        prompt = f"""[최근 대화 데이터]
{recent_chat}
[대상 참여자] {nickname}
[안내 유형] {level_guide.get(encouragement_level, level_guide[1])}
[최근 안내 문장]
{recent_messages_text}
{ENCOURAGEMENT_RULES}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=150,
            top_p=0.95,
            frequency_penalty=0.6,
            presence_penalty=0.6,
        )

        message = response.choices[0].message.content.strip()

        # 생성된 멘트를 히스토리에 추가 (최대 10개 유지)
        self.recent_messages.append(message)
        if len(self.recent_messages) > 10:
            self.recent_messages.pop(0)

        print(f"[GPT 독려 멘트] {nickname}님께: {message}")
        return message

    def _generate_template_encouragement(self, nickname: str, encouragement_level: int) -> str:
        """템플릿 기반 독려 멘트 생성 (다양성 확대)"""
        templates = {
            1: [
                f"{nickname}님, 지금 주제에 대한 생각도 듣고 싶어요! 😊 편하게 한마디 나눠주실래요?",
                f"{nickname}님, 영상에서 가장 눈에 들어온 장면은 무엇이었나요? 💡",
                f"{nickname}님, 다른 의견을 듣고 새롭게 떠오른 생각이 있다면 들려주세요. ✨",
                f"{nickname}님, 영상 속 선택 중 공감되는 부분이 있었나요? 🙂",
                f"{nickname}님, 아직 의견이 정리되지 않았다면 궁금한 점부터 꺼내주셔도 좋아요. 💬",
                f"{nickname}님, 지금 주제에서 함께 더 이야기해 보고 싶은 부분이 있나요? 😊",
                f"{nickname}님, 다른 관점으로 볼 수 있는 부분이 있다면 함께 나눠주세요. 💡",
                f"{nickname}님, 영상 속 인물에게 한 가지 물어볼 수 있다면 어떤 질문을 하고 싶으세요? 🤔",
                f"{nickname}님, 지금 논의에서 기억해 두고 싶은 내용을 하나 골라볼까요? ✨",
                f"{nickname}님, 의견과 질문 모두 환영이에요! 준비되시면 함께 이야기 나눠요. 😊",
            ],
            2: [
                f"{nickname}님, 지금 주제와 관련해 영상 속 선택의 이유를 어떻게 보셨나요? 🤔",
                f"{nickname}님, 영상 속 인물이 결정하기 전에 먼저 확인하면 좋을 점은 무엇일까요? 💡",
                f"{nickname}님, 이야기 나온 선택지 중 비교해 보고 싶은 두 가지가 있나요? 🙂",
                f"{nickname}님, 현재 주제에서 가장 중요하게 볼 기준을 하나 꼽는다면 무엇일까요? ✨",
                f"{nickname}님, 영상 속 인물에게 다른 선택지도 있을지 함께 생각해 볼까요? 💬",
                f"{nickname}님, 다른 학생의 의견 중 공감되는 부분과 그 이유가 궁금해요. 😊",
                f"{nickname}님, 지금 논의한 방법을 실행하기 전에 확인할 조건은 무엇일까요? 💡",
                f"{nickname}님, 영상 속 인물이 놓쳤을 수 있는 점이 있다면 편하게 들려주세요. 🤔",
                f"{nickname}님, 현재 주제에서 장점과 주의할 점을 함께 짚어볼까요? 🙂",
                f"{nickname}님, 영상 속 인물에게 작은 행동 하나를 제안한다면 무엇이 좋을까요? ✨",
            ],
            3: [
                f"{nickname}님, 지금 이야기에서 궁금한 부분이 있으면 편하게 물어봐 주세요. 😊",
                f"{nickname}님, 생각을 정리할 시간이 필요하면 천천히 듣고 계셔도 괜찮아요. 🌿",
                f"{nickname}님, 함께 다시 살펴보고 싶은 영상 장면이 있나요? 💬",
                f"{nickname}님, 낯선 용어가 있다면 그 단어부터 함께 이야기해 봐요. 💡",
                f"{nickname}님, 긴 설명 없이 생각나는 단어나 짧은 질문으로 시작해도 좋아요. 🙂",
                f"{nickname}님, 개인 경험을 말하지 않고 영상 속 상황으로 이야기하셔도 괜찮아요. 😊",
                f"{nickname}님, 서로 다른 의견 중 더 설명을 듣고 싶은 부분이 있나요? 🤔",
                f"{nickname}님, 글로 정리하기 어렵다면 어떤 부분이 고민되는지만 알려주셔도 좋아요. 💬",
                f"{nickname}님, 지금 주제와 관련해 교사에게 확인하고 싶은 점이 있나요? 💡",
                f"{nickname}님, 정답을 정하기보다 떠오른 생각을 함께 살펴보는 시간이니 편하게 참여해 주세요. ✨",
            ],
        }

        messages = templates.get(encouragement_level, templates[1])

        # 중복 방지: 최근 사용된 템플릿 제외
        available_messages = [msg for msg in messages if msg not in self.recent_messages[-5:]]

        # 모든 메시지가 최근에 사용되었다면 전체 풀에서 선택
        if not available_messages:
            available_messages = messages

        message = random.choice(available_messages)

        # 생성된 멘트를 히스토리에 추가
        self.recent_messages.append(message)
        if len(self.recent_messages) > 10:
            self.recent_messages.pop(0)

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
