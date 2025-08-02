
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import Dict, Optional, List
from app.question_generator import QuestionGenerator
from app.message_classifier import MessageClassifier

load_dotenv()

app = FastAPI(
    title="CJ AI API", 
    version="1.0.0",
    description="CJ 식음 서비스 매니저 교육용 AI API - 토론 참여 분석 및 질문 생성"
)

# 초기화
question_generator = QuestionGenerator()
message_classifier = MessageClassifier()

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
    primary_trait: str
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

@app.get("/")
async def root():
    return {
        "message": "CJ AI API",
        "version": "1.0.0",
        "description": "CJ 식음 서비스 매니저 교육용 AI API",
        "endpoints": {
            "/question": "토론 참여 유도 질문 생성",
            "/classify": "메시지 CJ 인재상 분류",
            "/profile": "사용자 종합 프로필 생성"
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
            request.user_id, 
            request.context
        )
        
        return ClassifyResponse(
            cj_values=result["cj_values"],
            primary_trait=result["primary_trait"],
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"CJ AI API 서버 시작 중...")
    print(f"주소: http://{host}:{port}")
    print(f"API 문서: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)