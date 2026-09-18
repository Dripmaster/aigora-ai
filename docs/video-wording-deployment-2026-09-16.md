# AI 영상 표현 운영 반영 — 2026-09-16

- 버전: `financial-education-2026-09-16-video-wording-v3-draft`
- 질문·답변·참여 안내의 시나리오 표현을 영상 내용·영상 속 상황·영상 속 인물로 변경했다. 공통 역할 지침을 사용하는 분류·평가·요약에도 같은 표현 지침을 적용했다.
- 질문/답변 입력의 영상 내용 제목, 영상 스크립트 제목, 참여 안내 폴백, 기록 없는 개인 피드백을 함께 수정했다.
- API 필드, 4축 분류·점수 규칙, 영상 데이터의 내부 `scenario` 키는 유지했다.
- 변경 파일: app/financial_profile.py, app/financial_prompts.py, app/question_generator2.py, app/answer_generator.py, app/participant_monitor.py, app/discussion_evaluator.py.
- 대상: EC2 `/srv/financial-education/ai`, `financial-ai.service` 재시작. Node 서비스는 재시작하지 않았다.
- 백업: `/var/backups/financial-education/video-wording-20260916-vOD1bh/ai-before.tar.gz`
- 검증: 금융교육 자동 테스트 30개 통과. 변경 파일 6개 SHA256 로컬/운영 일치. 운영 health 새 버전/status ok, financial-ai/financial-node active. 배포된 공통/기능별 지침, 4차시 영상 입력, 기본 피드백 문구 확인.
- 추가 무기록 evaluate API 점검은 기존 입력 검증에 따라 HTTP 400을 반환했다. 해당 피드백은 배포 모듈 직접 호출로 확인했다.
- 실제 모델 호출 및 출력 품질 검수는 이번 배포에서 수행하지 않았다.
- 기존 금융교육 AI 작업본과 함께 로컬 미커밋 상태이며, 이번 운영 반영은 변경 파일 직접 배포로 수행했다.
