from app.financial_prompts import ANSWER_RULES
from app.financial_profile import FACILITATOR_PROMPT, lesson_text, video_script
from typing import Dict, List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import json 

load_dotenv()

class AnswerGenerator:

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, timeout=20.0, max_retries=0)
            self.gpt_enabled = True
            self.model = "gpt-4o-mini"
            print(f"AnswerGenerator: OpenAI API 키 설정 완료")
        else:
            self.gpt_enabled = False
            print("AnswerGenerator: OpenAI API 키 설정 실패 - 템플릿 모드로 작동")

        # 교육 컨텐츠 저장소
        self.educational_data = {}  # educational_content.json 전체 데이터
        self.training_content = {}  # 슬라이드 내용
        self.video_topics = {}      # 영상 주제
        self.video_details = {}     # 영상 상세 내용

        # AI 페르소나: 숙련된 토론 퍼실리테이터
        self.facilitator_persona = FACILITATOR_PROMPT

        self.system_prompt = self.facilitator_persona + '\n' + ANSWER_RULES

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

    # ========== 답변 생성 핵심 메서드 ==========

    def build_answer_prompt(self, nickname: str, discussion_topic: str,
                            video_script: str, slide_content: str,question_text:str,
                            chat_history: List[Dict]) -> str:
        """
        프롬프트 생성 - 답변 중심

        Args:
            nickname: 답변 대상 참여자 닉네임
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

        prompt = f"""[현재 토론 주제] {discussion_topic}
[대상 참여자] {nickname}
[영상 내용]
{video_script}
[현재 차시 자료 및 보충 교재]
{slide_content}
[최근 대화]
{chat_summary}
[참여자 질문]
{question_text}
{ANSWER_RULES}"""

        return prompt

    def generate_answer(self, nickname: str, discussion_topic: str,
                        video_script: str, slide_content: str,question_text:str,
                        chat_history: List[Dict]) -> str:
        """
        답변 생성 메인 메서드

        Args:
            nickname: 답변 대상(멘션) 참여자 닉네임
            discussion_topic: 현재 토론 주제
            video_script: 현재 토론 중인 영상의 스크립트
            slide_content: 슬라이드 내용
            chat_history: 실시간 채팅 내역

        Returns:
            생성된 답변 문자열
        """
        # GPT 답변 생성 시도
        if self.gpt_enabled:
            try:
                prompt = self.build_answer_prompt(
                    nickname, discussion_topic, video_script,
                    slide_content, question_text,chat_history
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400,
                    top_p=0.9,
                    frequency_penalty=0.4,
                    presence_penalty=0.4
                )

                answer = response.choices[0].message.content.strip()
                print(f"[GPT 답변 생성] {nickname}님께: {answer}")
                return answer

            except Exception as e:
                print(f"AI 생성 실패: {type(e).__name__}")
                return self._generate_fallback_answer(nickname)

        # 템플릿 기반 폴백
        return self._generate_fallback_answer(nickname)

    def _generate_fallback_answer(self, nickname: str) -> str:
        """템플릿 기반 폴백 답변"""
        return f"{nickname}님, 지금은 AI 답변을 생성하지 못했습니다. 질문 내용을 강사에게 확인해 주세요."

    # --- Backward compatibility (deprecated) ---
    def build_context_prompt(self, nickname: str, discussion_topic: str,
                             video_script: str, slide_content: str,
                             chat_history: List[Dict]) -> str:
        """DEPRECATED: 질문 프롬프트 → 답변 프롬프트로 위임"""
        return self.build_answer_prompt(nickname, discussion_topic, video_script, slide_content, "", chat_history)

    def generate_question(self, nickname: str, discussion_topic: str,
                          video_script: str, slide_content: str,
                          chat_history: List[Dict]) -> str:
        """DEPRECATED: 질문 생성 → 답변 생성으로 위임"""
        return self.generate_answer(nickname, discussion_topic, video_script, slide_content, "", chat_history)

    def _generate_fallback_question(self, nickname: str) -> str:
        """DEPRECATED: 질문 템플릿 → 답변 템플릿으로 위임"""
        return self._generate_fallback_answer(nickname)
