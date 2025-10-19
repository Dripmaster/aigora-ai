"""
DiscussionSummarizer 테스트 파일

다양한 케이스로 주제별 연관성 점수와 객관적인 발언 요약 기능을 테스트합니다.

테스트 케이스:
1. 기본 케이스 - 모든 주제에 고르게 발언
2. 편중 케이스 - 특정 주제에만 집중 발언
3. 혼합 케이스 - 여러 주제가 섞인 발언
"""

import json
from app.discussion_summarizer import DiscussionSummarizer


def create_test_case_1():
    """케이스 1: 모든 주제에 고르게 발언 (이상적인 케이스)"""
    discussion_topics = [
        {
            "name": "메뉴 개발 및 품질 관리",
            "description": "신메뉴 개발과 기존 메뉴 품질 유지"
        },
        {
            "name": "직원 교육 및 역량 강화",
            "description": "직원 교육 프로그램 및 스킬 향상"
        },
        {
            "name": "매장 환경 및 분위기",
            "description": "매장 인테리어, 청결도, 분위기 개선"
        },
        {
            "name": "디지털 전환",
            "description": "키오스크, 모바일 주문 등 디지털화"
        }
    ]

    chat_history = [
        # 메뉴 개발 관련 (김주임 4개)
        {"nickname": "박매니저", "text": "신메뉴 개발에 대한 의견을 들어볼까요?", "timestamp": "2025-01-20T09:00:00"},
        {"nickname": "김주임", "text": "계절 메뉴를 분기별로 출시하면 고객들이 신선함을 느낄 것 같습니다.", "timestamp": "2025-01-20T09:01:00"},
        {"nickname": "김주임", "text": "로컬 식재료를 활용한 메뉴 개발도 트렌드에 맞을 것 같아요.", "timestamp": "2025-01-20T09:02:00"},
        {"nickname": "이사원", "text": "비건 메뉴도 추가하면 좋겠습니다.", "timestamp": "2025-01-20T09:03:00"},
        {"nickname": "김주임", "text": "메뉴의 플레이팅도 중요합니다. SNS에 올릴 만한 비주얼이 필요해요.", "timestamp": "2025-01-20T09:04:00"},
        {"nickname": "김주임", "text": "정기적인 시식회를 통해 메뉴 품질을 검증하는 시스템이 필요합니다.", "timestamp": "2025-01-20T09:05:00"},

        # 직원 교육 관련 (김주임 3개)
        {"nickname": "박매니저", "text": "직원 교육은 어떻게 진행하면 좋을까요?", "timestamp": "2025-01-20T09:06:00"},
        {"nickname": "김주임", "text": "신입 직원을 위한 체계적인 온보딩 프로그램이 필수입니다.", "timestamp": "2025-01-20T09:07:00"},
        {"nickname": "이사원", "text": "서비스 매뉴얼을 만들면 좋겠어요.", "timestamp": "2025-01-20T09:08:00"},
        {"nickname": "김주임", "text": "월 1회 전 직원 대상 서비스 교육과 위생 교육을 진행해야 합니다.", "timestamp": "2025-01-20T09:09:00"},
        {"nickname": "김주임", "text": "우수 직원 인센티브 제도를 도입하면 동기부여가 될 것입니다.", "timestamp": "2025-01-20T09:10:00"},

        # 매장 환경 관련 (김주임 3개)
        {"nickname": "박매니저", "text": "매장 환경 개선 아이디어가 있나요?", "timestamp": "2025-01-20T09:11:00"},
        {"nickname": "김주임", "text": "조명을 따뜻한 색으로 바꾸면 분위기가 더 아늑해질 것 같습니다.", "timestamp": "2025-01-20T09:12:00"},
        {"nickname": "이사원", "text": "좌석 간격을 넓히면 프라이버시가 좋아질 것 같아요.", "timestamp": "2025-01-20T09:13:00"},
        {"nickname": "김주임", "text": "매장 음악도 시간대별로 다르게 틀면 좋겠어요. 점심에는 밝게, 저녁에는 차분하게.", "timestamp": "2025-01-20T09:14:00"},
        {"nickname": "김주임", "text": "화장실과 주방을 항상 청결하게 유지하는 것이 고객 만족의 기본입니다.", "timestamp": "2025-01-20T09:15:00"},

        # 디지털 전환 관련 (김주임 4개)
        {"nickname": "박매니저", "text": "디지털 시스템 도입에 대해 어떻게 생각하시나요?", "timestamp": "2025-01-20T09:16:00"},
        {"nickname": "김주임", "text": "키오스크를 도입하면 주문 대기 시간을 줄일 수 있습니다.", "timestamp": "2025-01-20T09:17:00"},
        {"nickname": "이사원", "text": "모바일 선주문 시스템도 필요합니다.", "timestamp": "2025-01-20T09:18:00"},
        {"nickname": "김주임", "text": "QR코드로 메뉴판을 제공하면 종이 낭비도 줄이고 위생적입니다.", "timestamp": "2025-01-20T09:19:00"},
        {"nickname": "김주임", "text": "고객 데이터를 분석해서 맞춤형 프로모션을 제공하면 재방문율이 높아질 것입니다.", "timestamp": "2025-01-20T09:20:00"},
        {"nickname": "김주임", "text": "배달 앱 통합 관리 시스템으로 효율성을 높여야 합니다.", "timestamp": "2025-01-20T09:21:00"},

        # 무관한 발언
        {"nickname": "김주임", "text": "오늘 회의 시간이 길어지네요.", "timestamp": "2025-01-20T09:22:00"},
        {"nickname": "이사원", "text": "점심 뭐 먹을까요?", "timestamp": "2025-01-20T09:23:00"},
    ]

    return {
        "name": "케이스 1: 균형잡힌 토론 참여",
        "topics": discussion_topics,
        "chat_history": chat_history,
        "user_id": "김주임",
        "expected": {
            "메뉴 개발 및 품질 관리": {"count": 4, "score": ">= 0.7"},
            "직원 교육 및 역량 강화": {"count": 3, "score": ">= 0.6"},
            "매장 환경 및 분위기": {"count": 3, "score": ">= 0.6"},
            "디지털 전환": {"count": 4, "score": ">= 0.7"},
        }
    }


def create_test_case_2():
    """케이스 2: 한 주제에만 집중 발언 (편중 케이스)"""
    discussion_topics = [
        {"name": "고객 불만 처리", "description": "고객 불만 및 컴플레인 대응"},
        {"name": "배달 서비스 개선", "description": "배달 품질 및 시간 개선"},
        {"name": "프로모션 전략", "description": "마케팅 및 프로모션 기획"},
        {"name": "원가 관리", "description": "식자재 원가 절감 방안"}
    ]

    chat_history = [
        # 고객 불만 처리만 집중 (최대리 8개)
        {"nickname": "김부장", "text": "고객 불만에 어떻게 대응하고 계신가요?", "timestamp": "2025-01-21T10:00:00"},
        {"nickname": "최대리", "text": "고객 불만은 접수 즉시 처리하는 것이 원칙입니다.", "timestamp": "2025-01-21T10:01:00"},
        {"nickname": "최대리", "text": "불만 유형을 DB화해서 패턴을 분석하고 재발 방지 대책을 세워야 합니다.", "timestamp": "2025-01-21T10:02:00"},
        {"nickname": "박사원", "text": "고객 의견을 경청하는 자세가 중요하죠.", "timestamp": "2025-01-21T10:03:00"},
        {"nickname": "최대리", "text": "불만 고객에게는 즉시 사과하고 보상 기준을 명확히 해야 합니다.", "timestamp": "2025-01-21T10:04:00"},
        {"nickname": "최대리", "text": "컴플레인 처리 후 반드시 고객 만족도 확인 전화를 드려야 합니다.", "timestamp": "2025-01-21T10:05:00"},
        {"nickname": "최대리", "text": "직원들에게 고객 응대 매뉴얼을 숙지시켜 일관된 대응을 해야 합니다.", "timestamp": "2025-01-21T10:06:00"},
        {"nickname": "최대리", "text": "불만 사례를 월례 회의에서 공유하고 개선책을 함께 논의합니다.", "timestamp": "2025-01-21T10:07:00"},
        {"nickname": "최대리", "text": "SNS 리뷰 모니터링도 중요합니다. 부정적 댓글에 신속히 대응해야 합니다.", "timestamp": "2025-01-21T10:08:00"},
        {"nickname": "최대리", "text": "VIP 고객의 불만은 더욱 신중하게 처리하고 특별 관리가 필요합니다.", "timestamp": "2025-01-21T10:09:00"},

        # 다른 주제는 무관한 발언만
        {"nickname": "최대리", "text": "오늘 날씨가 쌀쌀하네요.", "timestamp": "2025-01-21T10:10:00"},
        {"nickname": "박사원", "text": "커피 한 잔 하실래요?", "timestamp": "2025-01-21T10:11:00"},
    ]

    return {
        "name": "케이스 2: 단일 주제 집중",
        "topics": discussion_topics,
        "chat_history": chat_history,
        "user_id": "최대리",
        "expected": {
            "고객 불만 처리": {"count": 8, "score": ">= 0.8"},
            "배달 서비스 개선": {"count": 0, "score": "= 0.0"},
            "프로모션 전략": {"count": 0, "score": "= 0.0"},
            "원가 관리": {"count": 0, "score": "= 0.0"},
        }
    }


def create_test_case_3():
    """케이스 3: 여러 주제가 섞인 발언 (복합 케이스)"""
    discussion_topics = [
        {"name": "지속가능 경영", "description": "친환경 운영 및 ESG"},
        {"name": "직원 복지", "description": "직원 처우 및 복지 제도"},
        {"name": "신기술 도입", "description": "AI, 자동화 등 신기술"},
    ]

    chat_history = [
        # 복합적 발언들 (정과장)
        {"nickname": "정과장", "text": "일회용품 사용을 줄이고 다회용기를 도입하면 환경에도 좋고 비용도 절감됩니다.", "timestamp": "2025-01-22T11:00:00"},
        {"nickname": "정과장", "text": "직원들의 근무 환경을 개선해야 이직률이 낮아집니다. 휴게실 확충이 필요합니다.", "timestamp": "2025-01-22T11:01:00"},
        {"nickname": "정과장", "text": "AI 챗봇으로 고객 문의를 자동 응대하면 직원 업무가 줄어들어요.", "timestamp": "2025-01-22T11:02:00"},
        {"nickname": "정과장", "text": "유기농 식재료를 사용하면 건강에도 좋고 친환경 이미지도 높아집니다.", "timestamp": "2025-01-22T11:03:00"},
        {"nickname": "정과장", "text": "직원 건강검진을 연 2회로 늘리고 복지포인트를 지급하면 만족도가 올라갑니다.", "timestamp": "2025-01-22T11:04:00"},
        {"nickname": "정과장", "text": "로봇 서빙 시스템을 도입하면 인건비를 절감하고 효율성을 높일 수 있습니다.", "timestamp": "2025-01-22T11:05:00"},
        {"nickname": "정과장", "text": "음식물 쓰레기 감량을 위해 소량 메뉴 옵션을 제공하는 것도 방법입니다.", "timestamp": "2025-01-22T11:06:00"},
        {"nickname": "정과장", "text": "주 4일 근무제를 시범 도입해서 직원 만족도를 높이는 것도 고려해볼 만합니다.", "timestamp": "2025-01-22T11:07:00"},
        {"nickname": "정과장", "text": "스마트 주방 시스템으로 조리 시간과 에너지를 최적화할 수 있습니다.", "timestamp": "2025-01-22T11:08:00"},
    ]

    return {
        "name": "케이스 3: 복합 주제 발언",
        "topics": discussion_topics,
        "chat_history": chat_history,
        "user_id": "정과장",
        "expected": {
            "지속가능 경영": {"count": 3, "score": ">= 0.6"},
            "직원 복지": {"count": 3, "score": ">= 0.6"},
            "신기술 도입": {"count": 3, "score": ">= 0.6"},
        }
    }


def run_test_case(summarizer, test_case):
    """단일 테스트 케이스 실행"""
    print("\n" + "=" * 80)
    print(f"테스트: {test_case['name']}")
    print("=" * 80)

    result = summarizer.summarize_user(
        user_id=test_case['user_id'],
        chat_history=test_case['chat_history'],
        discussion_topics=test_case['topics']
    )

    # 결과 출력
    print(f"\n사용자: {result['user_id']}")
    print(f"총 주제 수: {len(result['topics'])}")

    issues = []
    for topic_data in result['topics']:
        topic_name = topic_data['topic']
        score = topic_data['relevance_score']
        num_statements = len(topic_data['related_statements'])
        summary = topic_data['summary']

        print(f"\n📌 {topic_name}")
        print(f"   점수: {score:.2f} | 발언: {num_statements}개")

        # 기대값 확인
        expected = test_case['expected'].get(topic_name, {})
        expected_count = expected.get('count', 0)
        expected_score = expected.get('score', '')

        if num_statements != expected_count:
            print(f"   ⚠️  발언 수 불일치 (기대: {expected_count}, 실제: {num_statements})")
            issues.append(f"{topic_name}: 발언 수 {num_statements} != {expected_count}")

        # 요약문 체크
        has_summary = summary and summary != "이 주제와 관련된 발언이 없습니다."
        if num_statements > 0 and not has_summary:
            print(f"   ❌ 발언이 있는데 요약 없음!")
            issues.append(f"{topic_name}: 요약 누락")
        elif num_statements == 0 and has_summary:
            print(f"   ⚠️  발언이 없는데 요약 있음")
            issues.append(f"{topic_name}: 불필요한 요약")
        elif has_summary:
            print(f"   ✅ 요약: {summary[:50]}...")

    # 최종 판정
    print("\n" + "-" * 80)
    if not issues:
        print("✅ 테스트 통과!")
    else:
        print(f"❌ 테스트 실패 ({len(issues)}개 이슈)")
        for issue in issues:
            print(f"   - {issue}")

    return len(issues) == 0


def main():
    """메인 테스트 실행"""
    print("=" * 80)
    print("DiscussionSummarizer 종합 테스트")
    print("=" * 80)

    summarizer = DiscussionSummarizer()

    test_cases = [
        create_test_case_1(),
        create_test_case_2(),
        create_test_case_3(),
    ]

    results = []
    for test_case in test_cases:
        passed = run_test_case(summarizer, test_case)
        results.append((test_case['name'], passed))

    # 최종 결과
    print("\n" + "=" * 80)
    print("전체 테스트 결과")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n통과율: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")

    if passed_count == total_count:
        print("\n🎉 모든 테스트를 통과했습니다!")
    else:
        print(f"\n⚠️  {total_count - passed_count}개 테스트가 실패했습니다.")


if __name__ == "__main__":
    main()
