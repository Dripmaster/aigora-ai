
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi import Form
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import Dict, Optional, List
from app.question_generator import QuestionGenerator
from app.message_classifier2 import MessageClassifier2
from app.discussion_evaluator import PersonalEvaluator

load_dotenv()

app = FastAPI(
    title="CJ AI API", 
    version="1.0.0",
    description="CJ 식음 서비스 매니저 교육용 AI API - 토론 참여 분석 및 질문 생성"
)

# 초기화
question_generator = QuestionGenerator()
message_classifier = MessageClassifier2()
discussion_evaluator = PersonalEvaluator()

class QuestionRequest(BaseModel):
    user_id: str
    nickname: str 
    idle_time: int
    
class QuestionResponse(BaseModel):
    question: str
    target_user: str
   
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

@app.get("/")
async def root():
    return {
        "message": "CJ AI API",
        "version": "1.0.0",
        "description": "CJ 식음 서비스 매니저 교육용 AI API",
        "endpoints": {
            "/question": "토론 참여 유도 질문 생성",
            "/classify": "메시지 CJ 인재상 분류",
            "/profile": "사용자 종합 프로필 생성",
            "/evaluate": "개인 맞춤형 토론 총평 생성"
        }
    }

@app.post("/question", response_model = QuestionResponse)
async def question(request: QuestionRequest):
    # 질문 생성 컨텍스트 구성
    context = {
        "current_topic": "CJ 인재상 실천",
        "use_gpt": True
    }
    
    generated_question = question_generator.generate(
        request.user_id, 
        request.nickname, 
        request.idle_time, 
        context
    )
    
    return QuestionResponse(
        question=generated_question, 
        target_user=request.nickname
    )

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """메시지를 CJ 인재상 기준으로 분류"""
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
        evaluation_result = personal_evaluator.evaluate_user(
                     request.user_id,
                     request.user_messages,
                  request.discussion_context
        )    
        return EvaluationResponse(**evaluation_result)
             
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"총평 생성 중 오류 발생: {str(e)}")

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