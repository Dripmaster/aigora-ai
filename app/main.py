
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi import Form
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import Dict, Optional, List
from app.question_generator2 import QuestionGenerator2
from app.participant_monitor import ParticipantMonitor
from app.message_classifier2 import MessageClassifier2
from app.message_classifier_gpt import MessageClassifierGPT
from app.discussion_evaluator import PersonalEvaluator

load_dotenv()

app = FastAPI(
    title="CJ AI API",
    version="2.0.0",
    description="CJ 식음 서비스 매니저 교육용 AI API - 토론 참여 분석 및 질문 생성"
)

# 초기화
question_generator2 = QuestionGenerator2()
participant_monitor = ParticipantMonitor()
message_classifier = MessageClassifier2()
message_classifier_gpt = MessageClassifierGPT()
discussion_evaluator = PersonalEvaluator()

# QuestionGenerator2 교육 컨텐츠 로드
educational_content_path = os.path.join(os.path.dirname(__file__), "..", "educational_content.json")
if os.path.exists(educational_content_path):
    question_generator2.load_educational_content(educational_content_path)
else:
    print(f"경고: educational_content.json 파일을 찾을 수 없습니다: {educational_content_path}")

class Question2Request(BaseModel):
    nickname: str
    discussion_topic: str
    video_id: str
    chat_history: List[Dict]

class Question2Response(BaseModel):
    question: str
    target_user: str
    video_id: str
    discussion_topic: str

class EncouragementRequest(BaseModel):
    nickname: str
    chat_history: List[Dict]

class EncouragementResponse(BaseModel):
    nickname: str
    message: str

class ClassifyRequest(BaseModel):
    text: str
    user_id: str
    context: Optional[Dict] = None

class ClassifyResponse(BaseModel):
    cj_values: Dict[str, int]
    primary_trait: List[str]
    summary: str
    user_id: str

class ProfileRequest(BaseModel):
    user_id: str
    messages: List[Dict]

class ProfileResponse(BaseModel):
    user_id: str
    message_count: int
    avg_cj_values: Dict[str, int]
    top_traits: List[str]
    overall_summary: str

class ClassifyGPTRequest(BaseModel):
    text: str
    user_id: str
    context: Optional[Dict] = None

class ClassifyGPTResponse(BaseModel):
    cj_values: Dict[str, int]
    primary_trait: str
    summary: str
    user_id: str

class EvaluationRequest(BaseModel):
    user_id: str
    user_messages: List[Dict]
    discussion_context: Optional[Dict] = None

class EvaluationResponse(BaseModel):
    user_id: str
    overall_score: int
    cj_trait_scores: Dict[str, int]
    participation_summary: str
    strengths: List[str]
    improvements: List[str]
    personalized_feedback: str
    top_messages: List[str]
    evaluation_date: str
    message_count: int
    evaluation_method: str

class DiscussionOverallRequest(BaseModel):
    all_user_messages: List[Dict]
    discussion_context: Optional[Dict] = None

class DiscussionOverallResponse(BaseModel):
    discussion_summary: str
    total_participants: int
    total_messages: int
    evaluation_date: str
    evaluation_method: str

@app.get("/")
async def root():
    return {
        "message": "CJ AI API",
        "version": "2.0.0",
        "description": "CJ 식음 서비스 매니저 교육용 AI API",
        "endpoints": {
            "/question": "토론 참여 유도 질문 생성 (교육 컨텐츠 기반)",
            "/encouragement": "참여 독려 멘트 생성",
            "/classify": "메시지 CJ 인재상 분류 (규칙 기반)",
            "/classify-gpt": "메시지 CJ 인재상 분류 (GPT 기반)",
            "/profile": "사용자 종합 프로필 생성",
            "/evaluate": "개인 맞춤형 토론 총평 생성",
            "/discussion-overall": "전체 토론 AI 총평 생성"
        }
    }

@app.post("/question", response_model=Question2Response)
async def question(request: Question2Request):
    """
    교육 컨텐츠 기반 토론 참여 유도 질문 생성 (QuestionGenerator2)

    입력:
    - nickname: 질문 대상 참여자 닉네임
    - discussion_topic: 현재 토론 주제
    - video_id: 현재 토론 중인 영상 ID (예: "video_tous_1")
    - chat_history: 실시간 채팅 내역 [{"nickname": "김매니저", "text": "..."}, ...]
    """
    try:
        # 입력 검증
        if not request.nickname:
            raise HTTPException(status_code=400, detail="닉네임이 필요합니다")

        if not request.video_id:
            raise HTTPException(status_code=400, detail="영상 ID가 필요합니다")

        # 영상 스크립트 가져오기
        video_script = question_generator2.get_video_script(request.video_id)

        if "찾을 수 없습니다" in video_script:
            raise HTTPException(
                status_code=404,
                detail=f"영상 ID '{request.video_id}'를 찾을 수 없습니다"
            )

        # 슬라이드 내용 가져오기
        slide_content = question_generator2.get_slide_content_text()

        # 질문 생성
        generated_question = question_generator2.generate_question(
            nickname=request.nickname,
            discussion_topic=request.discussion_topic,
            video_script=video_script,
            slide_content=slide_content,
            chat_history=request.chat_history
        )

        return Question2Response(
            question=generated_question,
            target_user=request.nickname,
            video_id=request.video_id,
            discussion_topic=request.discussion_topic
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"질문 생성 중 오류 발생: {str(e)}"
        )

@app.post("/encouragement", response_model=EncouragementResponse)
async def generate_encouragement(request: EncouragementRequest):
    """
    참여 독려 멘트 생성

    입력:
    - nickname: 독려 대상 참여자 닉네임
    - chat_history: 실시간 채팅 내역 [{"nickname": "김매니저", "text": "..."}, ...]

    출력:
    - nickname: 대상 참여자
    - message: 생성된 독려 멘트
    """
    try:
        # 입력 검증
        if not request.nickname:
            raise HTTPException(status_code=400, detail="닉네임이 필요합니다")

        # 독려 레벨 1 (부드러운 초대) 고정
        encouragement_level = 1

        # 독려 멘트 생성
        message = participant_monitor.generate_encouragement_message(
            nickname=request.nickname,
            chat_history=request.chat_history,
            encouragement_level=encouragement_level
        )

        return EncouragementResponse(
            nickname=request.nickname,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"독려 멘트 생성 중 오류: {str(e)}"
        )

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """메시지를 CJ 인재상 기준으로 분류 (규칙 기반)"""
    try:
        # 입력 검증
        if not request.text or len(request.text.strip()) < 1:
            raise HTTPException(status_code=400, detail="메시지가 비어있습니다")

        if not request.user_id:
            raise HTTPException(status_code=400, detail="사용자 ID가 필요합니다")

        # 메시지 분류 수행
        result = message_classifier.classify(
            request.text,
            request.user_id
        )

        return ClassifyResponse(
            cj_values=result["cj_values"],
            primary_trait=result["multiple_traits"],
            summary=result["summary"],
            user_id=request.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분류 처리 중 오류 발생: {str(e)}")

@app.post("/classify-gpt", response_model=ClassifyGPTResponse)
async def classify_gpt(request: ClassifyGPTRequest):
    """메시지를 CJ 인재상 기준으로 분류 (GPT 기반)"""
    try:
        # 입력 검증
        if not request.text or len(request.text.strip()) < 1:
            raise HTTPException(status_code=400, detail="메시지가 비어있습니다")

        if not request.user_id:
            raise HTTPException(status_code=400, detail="사용자 ID가 필요합니다")

        # GPT 메시지 분류 수행
        result = message_classifier_gpt.classify(
            request.text,
            request.user_id,
            request.context
        )

        return ClassifyGPTResponse(
            cj_values=result["cj_values"],
            primary_trait=result["primary_trait"],
            summary=result["summary"],
            user_id=request.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT 분류 처리 중 오류 발생: {str(e)}")

@app.post("/profile", response_model=ProfileResponse)
async def profile(request: ProfileRequest):
    """사용자의 종합 프로필 생성"""
    try:
        # 입력 검증
        if not request.user_id:
            raise HTTPException(status_code=400, detail="사용자 ID가 필요합니다")
        
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="분석할 메시지가 없습니다")
        
        # 사용자 프로필 생성
        profile_result = message_classifier.get_user_profile(
            request.user_id,
            request.messages
        )
        
        # 오류 처리
        if "error" in profile_result:
            raise HTTPException(status_code=400, detail=profile_result["error"])
        
        return ProfileResponse(
            user_id=profile_result["user_id"],
            message_count=profile_result["message_count"],
            avg_cj_values=profile_result["avg_cj_values"],
            top_traits=profile_result["top_traits"],
            overall_summary=profile_result["overall_summary"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프로필 생성 중 오류 발생: {str(e)}")
    
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_user(request: EvaluationRequest):
    """개별 사용자의 토론 참여 총평 생성"""
    try:
        # 입력 검증
        if not request.user_id:
            raise HTTPException(status_code=400, detail="사용자 ID가 필요합니다")
                 
        if not request.user_messages:
            raise HTTPException(status_code=400, detail="분석할 메시지가 없습니다")
                 
        # 개인 맞춤형 토론 총평 생성
        evaluation_result = discussion_evaluator.evaluate_user(
                     request.user_id,
                     request.user_messages,
                  request.discussion_context
        )    
        return EvaluationResponse(**evaluation_result)
             
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"총평 생성 중 오류 발생: {str(e)}")

@app.post("/discussion-overall", response_model=DiscussionOverallResponse)
async def discussion_overall(request: DiscussionOverallRequest):
    """전체 토론 참여자들의 토론 AI 총평 생성"""
    try:
        # 입력 검증
        if not request.all_user_messages or len(request.all_user_messages) == 0:
            raise HTTPException(status_code=400, detail="분석할 토론 메시지가 없습니다")
        
        # 전체 토론 AI 총평 생성
        evaluation_result = discussion_evaluator.evaluate_discussion_overall(
            request.all_user_messages,
            request.discussion_context
        )
        
        return DiscussionOverallResponse(**evaluation_result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"토론 총평 생성 중 오류 발생: {str(e)}")

@app.get("/form", response_class=HTMLResponse)
async def form_page():
    html_content = """
    <html>
        <head><title>CJ AI 테스트 폼</title></head>
        <body>
            <h2>CJ 인재상 메시지 분류 테스트</h2>
            <form action="/form/result" method="post">
                <label>사용자 ID:</label><br>
                <input type="text" name="user_id" required><br><br>
                <label>메시지:</label><br>
                <textarea name="text" rows="4" cols="50" required></textarea><br><br>
                <input type="submit" value="분석하기">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/form/result", response_class=HTMLResponse)
async def form_result(user_id: str = Form(...), text: str = Form(...)):
    try:
        request = ClassifyRequest(user_id=user_id, text=text)
        result = message_classifier.classify(request.text, request.user_id)

        html_content = f"""
        <html>
            <head><title>분석 결과</title></head>
            <body>
                <h2>분석 결과</h2>
                <p><strong>사용자 ID:</strong> {user_id}</p>
                <p><strong>주된 가치:</strong> {result['multiple_traits']}</p>
                <p><strong>요약:</strong> {result['summary']}</p>
                <p><strong>전체 값:</strong> {result['cj_values']}</p>
                <br><a href="/form">다시 분석하기</a>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    except Exception as e:
        return HTMLResponse(content=f"<h3>오류 발생: {str(e)}</h3>")
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"CJ AI API 서버 시작 중...")
    print(f"주소: http://{host}:{port}")
    print(f"API 문서: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)