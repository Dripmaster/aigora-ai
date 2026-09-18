from app.financial_prompts import QUESTION_RULES
from app.financial_profile import FACILITATOR_PROMPT, lesson_text, video_script
from typing import Dict, List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import json 

load_dotenv()

class QuestionGenerator2:

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, timeout=20.0, max_retries=0)
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

        # 중복 방지를 위한 최근 생성된 질문 히스토리
        self.recent_questions = []  # 최근 생성된 질문들 (최대 10개 유지)

        # AI 페르소나: 숙련된 토론 퍼실리테이터
        self.facilitator_persona = FACILITATOR_PROMPT

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
        return video_script(self.educational_data, video_id)

    def get_slide_content_text(self, video_id: str) -> str:
        return lesson_text(self.educational_data, video_id)

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
            slide_content: 현재 차시 슬라이드 및 보충 자료
            chat_history: 실시간 채팅 내역 [{"nickname": "참여자", "text": "저는..."}, ...]
        """

        # 전체 채팅 내용 파악
        chat_summary = ""
        if chat_history:
            recent_count = min(10, len(chat_history))  # 최근 10개 메시지
            chat_summary = f"**토론 내역 (최근 {recent_count}개 메시지):**\n"
            for msg in chat_history[-recent_count:]:
                chat_summary += f"- {msg.get('nickname', '참여자')}: {msg.get('text', '')}\n"
        else:
            chat_summary = "**토론 내역:** 제공된 대화 기록이 없습니다.\n"

        # 최근 생성된 질문 히스토리 추가 (중복 방지)
        recent_questions_text = ""
        if self.recent_questions:
            recent_questions_text = "\n**최근 생성된 질문 (중복 방지):**\n"
            for q in self.recent_questions[-5:]:  # 최근 5개만
                recent_questions_text += f"- {q}\n"
            recent_questions_text += "\n최근 질문과 의미가 같은 질문은 반복하지 마세요.\n"

        prompt = f"""[대상 참여자] {nickname}
[현재 토론 주제] {discussion_topic}
[영상 내용]
{video_script}
[현재 차시 자료 및 보충 교재]
{slide_content}
[최근 대화]
{chat_summary}
[최근 질문]
{recent_questions_text}
{QUESTION_RULES}"""

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
                    temperature=0.7,
                    max_tokens=200,  # 150 -> 200 (JSON 응답 충분히 수용)
                    response_format={"type": "json_object"}  # JSON 형식 강제
                )

                content = response.choices[0].message.content.strip()

                result = json.loads(content)
                if not isinstance(result, dict) or result.get('need_question') is not True:
                    return '결과없음'
                question = result.get('question')
                if not isinstance(question, str) or not question.strip():
                    return '결과없음'
                question = question.strip()
                self.recent_questions.append(question)
                self.recent_questions = self.recent_questions[-10:]
                return question

            except Exception as e:
                print(f"AI 생성 실패: {type(e).__name__}")
                return self._generate_fallback_question(nickname)

        # 템플릿 기반 폴백
        return self._generate_fallback_question(nickname)

    def _generate_fallback_question(self, nickname: str) -> str:
        """Do not invent a context-free interruption when generation fails."""
        return '결과없음'
