# 금융교육 AI API 키 교체 및 실호출 확인 — 2026-09-18

사용자가 로컬 파일로 전달한 새 OpenAI API 키를 금융교육 전용 EC2의 `/etc/financial-education/ai.env`에 반영하고 `financial-ai`를 재시작했다. `financial-ai`, `financial-node` 모두 active다. 새 키로 실제 모델 응답을 받았으며, 기존 크레딧 부족 오류는 이번 호출에서 발생하지 않았다. 잔액과 계정 소유자는 조회하지 않았다.

## 확인 결과

| 범위 | 결과 |
|---|---|
| 로컬 API + 실제 OpenAI | `/classify-gpt`, `/qa`, `/question`, `/evaluate`, `/discussion-overall`, `/user-summary`, `/encouragement`, `/profile`, `/classify` 9개 HTTP 200 및 응답 확인 |
| 코드 계약 검사 | `python -m unittest discover -s tests -p 'test_financial*.py'` 31개 통과 |
| 답변 보완 후 로컬 실제 OpenAI | 자유적금 관련 질문 3개에서 납입과 출금을 구분하는 응답 확인 |
| 운영 서버 + 실제 OpenAI | `/classify-gpt` 1회 assessed, `/qa` 2회 실제 답변 확인. 모두 HTTP 200 |
| 운영 프로필 | `financial-education-2026-09-18-savings-v6` |
| 배포 파일 | `app/financial_profile.py`, `app/financial_prompts.py`의 로컬/서버 SHA-256 일치 |

운영 확인은 AI 서버의 loopback API에 가상 발언을 보냈다. 학생 기록 및 DB에 테스트 데이터를 쓰지 않았다. 브라우저부터 Node를 거치는 전체 수업 리허설이나 20명 동시 실제 모델 부하는 이번 확인 범위가 아니다. 단일 사례 호출 성공은 모든 AI 답변의 정확성을 보장하지 않는다.

## 실제 응답에서 발견한 설명 보완

초기 답변이 자유적금을 자유롭게 입출금할 수 있는 상품으로 설명했다. 공통 역할과 질문 답변 지침에 납입의 자유와 출금 조건을 구별하도록 명시했다. 중도인출 여부와 중도해지 이율은 상품 약관을 확인하도록 했다. 공통 지침만 추가한 첫 재확인에서도 오답이 나와, 답변 지침에 잘못된 전제를 바로잡는 예시를 추가한 후 다시 확인했다.

근거: [우리은행 예금·적금 기초 가이드](https://spot.wooribank.com/pot/Dream?__STEP=1&withyou=CQCCS0095)의 정기적금·자유적금 구분 및 만기 이전 해지 설명. 외부 검색은 이 오류 확인에만 사용했으며 서비스에 검색/RAG 기능을 추가하지 않았다.

운영 서버 확인 예시:
- 질문: 자유적금이면 넣은 돈을 아무 때나 꺼내 써도 이자를 그대로 받나요?
- 답변 첫 문장: “자유적금이라고 자유롭게 출금하거나 약정 이자를 그대로 받을 수 있는 것은 아닙니다.” 이어 상품별 중도인출 조건 확인을 안내했다.

## 운영 기록

- 새 키 원문과 일부 문자열을 문서·Git·화면 출력에 기록하지 않았다. 로컬 `.env.local`은 Git 제외 및 권한 0600 유지.
- 서버 환경 파일은 root 소유 0600 유지. 기존 키는 폐기하지 않았다.
- 환경 파일 변경 전 백업: `/var/backups/financial-education/api-key-20260918-timk8z3_` (root 전용).
- 프롬프트 변경 전 백업: `/var/backups/financial-education/savings-prompt-20260918-7y2qxq9o`.
- 실행 증거: `/Users/yeongminson/cj_edu_ai/ai_server/tmp/key-rotation-20260918/`의 `live-local.json`, `qa-regression-local-final.json`, `live-production.json`, `prompt-deploy-metadata.json`.
- 프롬프트는 EC2 서비스에 직접 반영했다. 이번 작업에서 웹 코드 변경이나 Vercel 재배포는 하지 않았다. AI 저장소의 이전 미커밋 변경은 유지했다.

남은 수업 준비는 영상 수령·적용과 교사/학생 전체 흐름 리허설이다.
