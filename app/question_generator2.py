import random
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
import json 

load_dotenv()

class QuestionGenerator2:

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.gpt_enabled = True
            self.model = "gpt-4o-mini"
            print(f"QuestionGenerator2: OpenAI API 키 설정 완료")
        else:
            self.gpt_enabled = False
            print("QuestionGenerator2: OpenAI API 키 설정 실패 - 템플릿 모드로 작동")

        # 교육 컨텐츠 저장소
        self.educational_data = {}  # educational_content.json 전체 데이터
        self.training_content = {}  # 슬라이드 내용
        self.video_topics = {}      # 영상 주제
        self.video_details = {}     # 영상 상세 내용

        # AI 페르소나: 숙련된 토론 퍼실리테이터
        self.facilitator_persona = """당신은 **CJ 식음 교육센터의 수석 토론 퍼실리테이터**입니다.

**경력 및 전문성:**
- 15년간 대기업 교육 프로그램 토론 진행 경험
- 10~30명 규모의 대규모 토론 운영 전문가
- 교육학 석사 및 퍼실리테이션 전문 자격 보유
- CJ 4대 가치(정직/열정/창의/존중)를 깊이 이해하고 실천

**토론 운영 철학:**
- "모든 참여자가 주인공입니다" - 한 명도 소외되지 않는 토론
- "경청과 공감이 먼저입니다" - 판단 전에 이해하기
- "실무 현장의 목소리를 담습니다" - 실천 가능한 해법 찾기
- "배움은 즐거워야 합니다" - 부담 없이 편안한 분위기

**토론 진행 스킬:**
- 발언 균형 조정: 과도한 발언자 조율, 조용한 참여자 격려
- 시간 관리: 주제별 시간 안배 및 흐름 조절
- 갈등 중재: 의견 충돌 시 건설적 방향 유도
- 깊이 있는 질문: 피상적 답변을 넘어 본질 탐구
- 즉각적 피드백: 긍정 강화 및 건설적 조언

**말투 특징:**
- 친근하면서도 전문성 있는 존댓말 (예: "~하시는군요", "~해주시면 좋을 것 같아요")
- 따뜻한 격려와 인정 (예: "좋은 지적이세요!", "그 경험 정말 소중하네요")
- 구체적이고 실천적인 질문 (예: "그때 어떤 감정이셨나요?", "다음엔 어떻게 하실 건가요?")
- 적절한 이모지 활용으로 친근감 UP (😊, 👏, 💡, ✨)

**교육 컨텐츠 활용:**
- 슬라이드 핵심 내용을 자연스럽게 질문에 녹이기
- 영상 속 사례를 토론 소재로 연결하기
- 이론과 실무를 연결하는 브릿지 질문 던지기
- 학습 목표 달성을 위한 전략적 질문 설계

**중요: 토론 내용 분석 기반 질문 생성**
- **AI가 토론 내용을 깊이 분석하고 있음을 드러내세요**
- 단순한 참여 유도가 아닌, 토론 흐름을 읽고 있다는 인상을 주세요
- 최근 나온 의견/키워드를 자연스럽게 언급하세요
- 토론의 방향성을 제시하는 분석적 질문을 하세요

**금기사항:**
- ❌ 딱딱하고 형식적인 말투
- ❌ 참여 강요나 압박
- ❌ 교육 내용과 무관한 질문
- ❌ 비난이나 부정적 피드백
- ❌ 토론 내용을 무시한 단순 참여 유도만 (예: "어떻게 생각하세요?" 같은 일반적 질문)

**목표:**
토론 내용을 깊이 분석하고 있음을 보여주며, 참여자가 자연스럽게 토론에 합류할 수 있는 맥락 있는 질문 제공
"""

        self.system_prompt = self.facilitator_persona

    # ========== 교육 컨텐츠 로딩 메서드 (JSON 파일 기반) ==========

    def load_educational_content(self, json_path: str):
        """
        educational_content.json 파일 로드

        Args:
            json_path: educational_content.json 파일 경로
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.educational_data = json.load(f)

            # 슬라이드 내용 로드
            self.training_content = self.educational_data.get("slide_content", {})

            # 비디오 내용 로드
            video_content = self.educational_data.get("video_content", {})
            for video_id, video_data in video_content.items():
                self.video_topics[video_id] = video_data.get("topic", "")
                self.video_details[video_id] = video_data.get("details", "")

            print(f"[컨텐츠 로드 완료] 슬라이드 {len(self.training_content)}개, 영상 {len(video_content)}개")
            return True

        except Exception as e:
            print(f"[컨텐츠 로드 실패] {e}")
            return False

    def get_video_script(self, video_id: str) -> str:
        """
        특정 비디오의 전체 스크립트 반환 (시나리오, 대화, 토론 질문 포함)

        Args:
            video_id: 비디오 ID (예: "video_tous_1")

        Returns:
            포맷팅된 비디오 스크립트
        """
        video_content = self.educational_data.get("video_content", {})
        if video_id not in video_content:
            return "영상 스크립트를 찾을 수 없습니다."

        video = video_content[video_id]

        # 스크립트 구성
        script = f"""제목: {video.get('topic', 'N/A')}
브랜드: {video.get('brand', 'N/A')}"""

        # role 필드가 있는 경우 추가
        if 'role' in video:
            script += f"\n직무: {video.get('role', 'N/A')}"

        script += f"""
핵심 가치: {', '.join(video.get('main_values', []))}

시나리오: {video.get('scenario', 'N/A')}

배경: {video.get('scene', 'N/A')}

등장인물: {video.get('characters', 'N/A')}

나레이션: {video.get('narration', 'N/A')}

대화 내용:
"""
        # 대화 추가 (텍스트 배열 형식)
        for dialogue in video.get('dialogue', []):
            script += f"{dialogue}\n"

        # 토론 질문 추가
        script += "\n토론 질문:\n"
        for idx, question in enumerate(video.get('discussion_questions', []), 1):
            script += f"{idx}. {question}\n"

        return script

    def get_slide_content_text(self) -> str:
        """
        슬라이드 내용을 하나의 텍스트로 반환

        Returns:
            포맷팅된 슬라이드 내용
        """
        slide_content = self.educational_data.get("slide_content", {})

        text = ""
        for slide_id, content in slide_content.items():
            text += f"### {slide_id}\n{content}\n\n"

        return text.strip()

    # ========== 질문 생성 핵심 메서드 ==========

    def build_context_prompt(self, nickname: str, discussion_topic: str,
                            video_script: str, slide_content: str,
                            chat_history: List[Dict]) -> str:
        """
        프롬프트 생성 - 새로운 입력 형식

        Args:
            nickname: 질문 대상 참여자 닉네임
            discussion_topic: 현재 토론 주제
            video_script: 현재 토론 중인 영상의 스크립트
            slide_content: 슬라이드 내용 (항상 같음)
            chat_history: 실시간 채팅 내역 [{"nickname": "김매니저", "text": "저는..."}, ...]
        """

        # 전체 채팅 내용 파악
        chat_summary = ""
        if chat_history:
            recent_count = min(10, len(chat_history))  # 최근 10개 메시지
            chat_summary = f"**토론 내역 (최근 {recent_count}개 메시지):**\n"
            for msg in chat_history[-recent_count:]:
                chat_summary += f"- {msg.get('nickname', '참여자')}: {msg.get('text', '')}\n"
        else:
            chat_summary = "**토론 내역:** 아직 토론이 시작되지 않았습니다.\n"

        prompt = f"""[토론 세션 정보]
**토론 주제:** {discussion_topic}

**영상 스크립트:**
{video_script}

**슬라이드 내용:**
{slide_content}

{chat_summary}

[질문 생성 미션]
{nickname}님에게 질문을 만들어주세요.

요구사항:
1. **교육 내용 연계**: 위 영상 스크립트와 슬라이드 내용을 자연스럽게 연결
2. **토론 흐름 고려**: 최근 채팅 내용을 읽고 맥락에 맞는 질문
3. **토론 분석 드러내기**: AI가 토론 내용을 깊이 분석하고 있음을 보여주는 질문
4. **친근한 톤**: 이모지 포함, 따뜻하고 격려하는 말투
5. **실천 중심**: 현장에서 적용 가능한 구체적 질문
6. **한 문장**: 간결하고 명확하게

**반드시 한국어로, 한 문장의 질문만 생성하세요.**"""

        return prompt

    def generate_question(self, nickname: str, discussion_topic: str,
                         video_script: str, slide_content: str,
                         chat_history: List[Dict]) -> str:
        """
        질문 생성 메인 메서드

        Args:
            nickname: 질문 대상 참여자 닉네임
            discussion_topic: 현재 토론 주제
            video_script: 현재 토론 중인 영상의 스크립트
            slide_content: 슬라이드 내용
            chat_history: 실시간 채팅 내역

        Returns:
            생성된 질문 문자열
        """
        # GPT 질문 생성 시도
        if self.gpt_enabled:
            try:
                prompt = self.build_context_prompt(
                    nickname, discussion_topic, video_script,
                    slide_content, chat_history
                )

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

                question = response.choices[0].message.content.strip()
                print(f"[GPT 질문 생성] {nickname}님께: {question}")
                return question

            except Exception as e:
                print(f"GPT API 오류: {e}, 템플릿 모드로 전환")
                return self._generate_fallback_question(nickname)

        # 템플릿 기반 폴백
        return self._generate_fallback_question(nickname)

    def _generate_fallback_question(self, nickname: str) -> str:
        """템플릿 기반 폴백 질문"""
        templates = [
            f"{nickname}님, 환영합니다! 😊 오늘 주제에 대해 어떻게 생각하시나요?",
            f"{nickname}님의 소중한 의견도 듣고 싶어요! ✨ 편하게 생각 나눠주시겠어요?",
            f"{nickname}님, 토론 주제 관련해서 경험이나 의견 있으시면 들려주세요! 👋",
            f"{nickname}님 생각도 궁금한데요! 💡 어떤 점이 인상적이셨나요?",
            f"{nickname}님, 혹시 비슷한 경험 있으셨나요? 😄 나눠주시면 좋을 것 같아요!"
        ]
        question = random.choice(templates)
        print(f"[템플릿 질문] {nickname}님께: {question}")
        return question
