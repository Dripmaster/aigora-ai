# AIGORA AI (CJ AI API)

CJ 식음 서비스 매니저 교육 토론을 지원하는 FastAPI 기반 백엔드 프로젝트입니다.
토론 참여 유도 질문 생성, 참여 독려, 메시지 분류, 개인/전체 총평, 사용자 주제별 요약 기능을 제공합니다.

## 1) 프로젝트 개요

이 프로젝트는 교육 현장에서 발생하는 실시간 채팅 데이터를 바탕으로 아래 기능을 제공합니다.

- **질문 생성**: 교육 컨텐츠(슬라이드/영상) + 최근 채팅 맥락을 반영해 참여 유도 질문 생성
- **답변 생성(QA)**: 질문 텍스트와 교육 컨텐츠를 반영해 답변 생성
- **참여 독려**: 저활동 참여자에게 자연스러운 독려 메시지 생성
- **CJ 인재상 분류**: `정직/열정/창의/존중` 기준으로 발언 분석
- **개인/전체 토론 총평**: 사용자 단위 및 토론 전체 단위 피드백 생성
- **사용자 주제 요약**: 지정한 토론 주제별 관련 발언 추출 + 관련도 점수 + 요약

## 2) 기술 스택

- **언어/런타임**: Python 3.9
- **웹 프레임워크**: FastAPI, Uvicorn
- **AI/LLM**: OpenAI API (`gpt-4o-mini` 중심)
- **NLP**: KoNLPy(Okt), Gensim(Word2Vec)
- **환경 관리**: `requirements.txt`, `environment.yml`, `.env`

## 3) 디렉터리 구조

```text
.
├── app/
│   ├── main.py                      # FastAPI 엔트리포인트/라우팅
│   ├── question_generator2.py       # 교육 컨텐츠 기반 질문 생성
│   ├── answer_generator.py          # 질문 기반 답변 생성 (/qa)
│   ├── participant_monitor.py       # 참여 독려 메시지 생성
│   ├── message_classifier2.py       # 규칙+유사도 기반 CJ 인재상 분류
│   ├── message_classifier_gpt.py    # GPT 기반 CJ 인재상 분류
│   ├── message_classifier.py        # 기존 분류기 및 프로필 계산
│   ├── discussion_evaluator.py      # 개인/전체 토론 총평
│   ├── discussion_summarizer.py     # 사용자 주제별 요약
│   └── question_generator.py        # 구 버전 질문 생성기(레거시)
├── models/
│   └── ko.bin                       # Word2Vec 모델 파일
├── educational_content.json         # 슬라이드/영상 교육 컨텐츠
├── requirements.txt                 # pip 의존성
├── environment.yml                  # conda 환경 정의
├── test_classifier2_scenarios.py    # 분류기 시나리오 테스트 스크립트
├── test_discussion_summarizer.py    # 요약기 테스트 스크립트
└── scenario*_discussion_cases.py    # 토론 시나리오 데이터
```

## 4) 빠른 시작

### 4.1 Python 가상환경 + pip 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 Conda 사용 시

```bash
conda env create -f environment.yml
conda activate cj-ai
```

### 4.3 환경 변수 설정 (`.env`)

프로젝트 루트에 `.env` 파일을 만들고 아래 값을 설정하세요.

```env
OPENAI_API_KEY=your_openai_api_key
HOST=0.0.0.0
PORT=8000

# discussion_summarizer.py 관련(선택)
DISCUSSION_SUMMARY_MAX_MESSAGES=60
DISCUSSION_SUMMARY_MODEL=gpt-4o-mini
DISCUSSION_SUMMARY_FAST=1
DISCUSSION_SUMMARY_CACHE_TTL=300
DISCUSSION_SUMMARY_CACHE_MAX=256
```

> `OPENAI_API_KEY`가 없으면 일부 기능은 템플릿/제한 모드로 동작하거나 오류가 발생할 수 있습니다.

## 5) 실행 방법

### 5.1 직접 실행

```bash
python app/main.py
```

### 5.2 Uvicorn 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

서버 실행 후 접속:

- API 루트: `http://localhost:8000/`
- Swagger 문서: `http://localhost:8000/docs`
- 테스트 폼: `http://localhost:8000/form`

## 6) 주요 API 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| POST | `/question` | 교육 컨텐츠 + 채팅 맥락 기반 질문 생성 |
| POST | `/qa` | 질문 텍스트 기반 답변 생성 |
| POST | `/encouragement` | 참여 독려 메시지 생성 |
| POST | `/classify` | 규칙/유사도 기반 CJ 인재상 분류 |
| POST | `/classify-gpt` | GPT 기반 CJ 인재상 분류 |
| POST | `/profile` | 사용자 종합 프로필 생성 |
| POST | `/evaluate` | 개인 토론 총평 생성 |
| POST | `/discussion-overall` | 전체 토론 총평 생성 |
| POST | `/user-summary` | 사용자 주제별 발언 요약/관련도 분석 |
| GET | `/form` | 웹 폼 기반 분류 테스트 페이지 |

## 7) 테스트/검증 스크립트

프로젝트에는 시나리오 기반 테스트 스크립트가 포함되어 있습니다.

```bash
python test_classifier2_scenarios.py
python test_discussion_summarizer.py
```

> OpenAI API 키, KoNLPy/JVM, Word2Vec 모델(`models/ko.bin`) 준비 상태에 따라 일부 테스트는 실패할 수 있습니다.

## 8) 구현 시 참고 사항

- `question_generator2`/`answer_generator`는 `educational_content.json`을 로드해 질문/답변 품질을 높입니다.
- `message_classifier2`는 KoNLPy 형태소 분석 + 키워드 매칭 + Word2Vec 유사도 점수를 결합합니다.
- `discussion_summarizer`는 캐시/메시지 제한 옵션으로 성능을 조절할 수 있습니다.
- 루트(` / `) 응답에 기능 소개 및 엔드포인트 요약이 포함되어 있습니다.

## 9) 권장 개발 흐름

1. `.env` 구성 (`OPENAI_API_KEY` 포함)
2. 의존성 설치
3. `uvicorn app.main:app --reload`로 실행
4. `/docs`에서 API 호출 검증
5. 시나리오 테스트 스크립트로 회귀 확인

---

필요하면 다음 단계로 `curl` 예제 요청/응답 샘플 섹션도 추가해드릴게요.
