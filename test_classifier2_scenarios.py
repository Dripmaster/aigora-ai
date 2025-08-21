"""
MessageClassifier2 시나리오 테스트
기존 시나리오 파일들을 사용하여 MessageClassifier2의 성능을 검증합니다.
"""

import sys
import os
from statistics import mean
from typing import Dict, List

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.getcwd())

from app.message_classifier2 import MessageClassifier2
from scenario1_discussion_cases import get_scenario1_discussion_cases
from scenario2_discussion_cases import get_scenario2_discussion_cases
from scenario3_discussion_cases import get_scenario3_discussion_cases
from scenario4_discussion_cases import get_scenario4_discussion_cases

class Classifier2ScenarioTester:
    """MessageClassifier2로 모든 시나리오 테스트"""
    
    def __init__(self, similarity_threshold=0.4):
        self.classifier = MessageClassifier2(similarity_threshold)
        self.results = {
            "total_tests": 0,
            "correct_predictions": 0,
            "scenarios": {},
            "method_stats": {
                "keyword_matching": 0,
                "word2vec_similarity": 0,
                "classification_failed": 0,
                "skip": 0,
                "analysis_failed": 0,
                "error": 0
            }
        }
    
    def test_scenario(self, scenario_name: str, test_cases: list, show_details: bool = False, show_test_sentences: bool = False
                      ) -> dict:
        """특정 시나리오 테스트"""
        print(f"\n{'='*20} {scenario_name} 테스트 {'='*20}")
        
        scenario_results = {
            "total": len(test_cases),
            "correct": 0,
            "trait_stats": {},
            "method_stats": {"keyword_matching": 0, "word2vec_similarity": 0, "classification_failed": 0, "skip": 0, "analysis_failed": 0, "error": 0},
            "errors": []
        }
        
        # 인재상별 통계 초기화
        for case in test_cases:
            trait = case["expected_primary"]
            if trait not in scenario_results["trait_stats"]:
                scenario_results["trait_stats"][trait] = {"total": 0, "correct": 0}
        
        print(f"총 {len(test_cases)}개 테스트 케이스")
        if show_test_sentences:
            print("-" * 80)
        
        for i, case in enumerate(test_cases):
            text = case["text"]
            expected = case["expected_primary"]
            
            try:
                # MessageClassifier2로 분류
                result = self.classifier.classify(text, f"{scenario_name}_{i}")
                predicted = result["primary_trait"]
                method = result.get("method", "unknown")
                confidence = result.get("confidence", 0.0)
            except Exception as e:
                print(f"❌ 분류 중 오류 발생 (케이스 {i+1}): {e}")
                print(f"   문장: {text}")
                result = {
                    "primary_trait": "분류오류",
                    "method": "error",
                    "confidence": 0.0,
                    "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0}
                }
                predicted = "분류오류"
                method = "error"
                confidence = 0.0
            
            # 통계 업데이트
            scenario_results["trait_stats"][expected]["total"] += 1
            if method in scenario_results["method_stats"]:
                scenario_results["method_stats"][method] += 1
            if method in self.results["method_stats"]:
                self.results["method_stats"][method] += 1
            
            if predicted == expected:
                scenario_results["correct"] += 1
                scenario_results["trait_stats"][expected]["correct"] += 1
                status = "✅"
            else:
                status = "❌"
                scenario_results["errors"].append({
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                    "method": method,
                    "confidence": confidence,
                    "scores": result["cj_values"]
                })
            
            # 모든 테스트 문장 출력
            if show_test_sentences:
                try:
                    print(f"{i+1:3d}. {text}")
                    print(f"     예상: {expected} → 결과: {predicted} {status}")
                    print(f"     방법: {method}, 신뢰도: {confidence:.3f}")
                    if predicted != expected:
                        print(f"     점수: {result['cj_values']}")
                    print()
                except Exception as e:
                    print(f"     ❌ 출력 오류: {e}")
                    print(f"     결과 데이터: {result}")
                    print()
            
            # 상세 출력 (처음 5개만)
            elif show_details and i < 5:
                print(f"{i+1:2d}. {text[:50]}...")
                print(f"    예상: {expected}, 결과: {predicted} {status}")
                print(f"    방법: {method}, 신뢰도: {confidence:.3f}")
                print(f"    점수: {result['cj_values']}")
                print()
        
        # 시나리오별 결과 출력
        accuracy = scenario_results["correct"] / scenario_results["total"] * 100
        print(f"\n{scenario_name} 정확도: {scenario_results['correct']}/{scenario_results['total']} = {accuracy:.1f}%")
        
        # 방법별 통계
        print(f"분류 방법별 통계:")
        for method, count in scenario_results["method_stats"].items():
            if count > 0:
                percentage = count / scenario_results["total"] * 100
                print(f"  {method}: {count}개 ({percentage:.1f}%)")
        
        # 인재상별 정확도
        print(f"인재상별 정확도:")
        for trait, stats in scenario_results["trait_stats"].items():
            trait_accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {trait}: {stats['correct']}/{stats['total']} = {trait_accuracy:.1f}%")
        
        return scenario_results
    
    def show_error_analysis(self, scenario_name: str, errors: list, max_errors: int = 3):
        """오분류 사례 분석"""
        if not errors:
            print(f"\n{scenario_name}: 모든 테스트 케이스가 정확히 분류되었습니다! 🎉")
            return
        
        print(f"\n{scenario_name} 오분류 사례 분석 (총 {len(errors)}개 중 {min(max_errors, len(errors))}개):")
        print("-" * 80)
        
        for i, error in enumerate(errors[:max_errors]):
            print(f"{i+1}. 문장: '{error['text']}'")
            print(f"   예상: {error['expected']} vs 예측: {error['predicted']}")
            print(f"   방법: {error['method']}, 신뢰도: {error['confidence']:.3f}")
            print(f"   점수: {error['scores']}")
            
            # 점수 차이 분석
            expected_score = error['scores'][error['expected']]
            predicted_score = error['scores'][error['predicted']]
            score_diff = predicted_score - expected_score
            print(f"   점수 차이: {error['predicted']}({predicted_score}) - {error['expected']}({expected_score}) = {score_diff:+d}")
            print()
    
    def run_all_tests(self, show_details: bool = False, show_test_sentences: bool = False):
        """모든 시나리오 테스트 실행"""
        print("🧪 MessageClassifier2 - 모든 시나리오 종합 테스트")
        print("=" * 80)
        print("KoNLPy + Word2Vec 기반 분류기로 4개 시나리오의 실제 토론 발언 분류 성능 검증")
        if show_test_sentences:
            print("📝 모든 테스트 문장과 결과를 출력합니다.")
        print()
        
        # 각 시나리오 테스트
        scenarios = [
            ("시나리오 1: 갓 구운 빵의 비밀", get_scenario1_discussion_cases()),
            ("시나리오 2: 알바생 혜경이의 열정", get_scenario2_discussion_cases()),
            ("시나리오 3: 신메뉴 개발 아이디어", get_scenario3_discussion_cases()),
            ("시나리오 4: 고객 불만 응대", get_scenario4_discussion_cases())
        ]
        
        total_correct = 0
        total_tests = 0
        
        for scenario_name, test_cases in scenarios:
            scenario_results = self.test_scenario(scenario_name, test_cases, show_details, show_test_sentences)
            
            self.results["scenarios"][scenario_name] = scenario_results
            total_correct += scenario_results["correct"]
            total_tests += scenario_results["total"]
        
        # 전체 결과 요약
        overall_accuracy = total_correct / total_tests * 100
        print("\n" + "="*80)
        print("📊 MessageClassifier2 전체 테스트 결과 요약")
        print("="*80)
        print(f"전체 정확도: {total_correct}/{total_tests} = {overall_accuracy:.1f}%")
        print()
        
        # 전체 방법별 통계
        print("전체 분류 방법별 통계:")
        for method, count in self.results["method_stats"].items():
            if count > 0:
                percentage = count / total_tests * 100
                print(f"  {method}: {count}개 ({percentage:.1f}%)")
        print()
        
        # 시나리오별 요약
        print("시나리오별 정확도:")
        for scenario_name, results in self.results["scenarios"].items():
            accuracy = results["correct"] / results["total"] * 100
            print(f"  {scenario_name}: {accuracy:.1f}% ({results['correct']}/{results['total']})")
        
        print()
        
        # 전체 인재상별 통계
        self.show_overall_trait_stats()
        
        # 각 시나리오별 오분류 분석
        print("\n" + "="*80)
        print("🔍 시나리오별 오분류 사례 분석")
        print("="*80)
        
        for scenario_name, results in self.results["scenarios"].items():
            self.show_error_analysis(scenario_name, results["errors"])
        
        # 성능 비교 및 분석
        self.show_performance_analysis(overall_accuracy, total_tests)
    
    def show_overall_trait_stats(self):
        """전체 인재상별 통계"""
        overall_trait_stats = {}
        
        # 모든 시나리오의 인재상별 통계 집계
        for scenario_results in self.results["scenarios"].values():
            for trait, stats in scenario_results["trait_stats"].items():
                if trait not in overall_trait_stats:
                    overall_trait_stats[trait] = {"total": 0, "correct": 0}
                overall_trait_stats[trait]["total"] += stats["total"]
                overall_trait_stats[trait]["correct"] += stats["correct"]
        
        print("전체 인재상별 정확도:")
        for trait, stats in overall_trait_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"] * 100
                print(f"  {trait}: {stats['correct']}/{stats['total']} = {accuracy:.1f}%")
    
    def show_performance_analysis(self, accuracy, total_tests):
        """성능 분석 및 개선 제안"""
        print("\n" + "="*80)
        print("🔍 MessageClassifier2 성능 분석")
        print("="*80)
        
        keyword_success = self.results["method_stats"]["keyword_matching"]
        word2vec_success = self.results["method_stats"]["word2vec_similarity"] 
        failed_classifications = self.results["method_stats"]["classification_failed"]
        
        keyword_rate = keyword_success / total_tests * 100
        word2vec_rate = word2vec_success / total_tests * 100
        failure_rate = failed_classifications / total_tests * 100
        
        print(f"📈 분류 성공률 분석:")
        print(f"  - 키워드 매칭 성공률: {keyword_rate:.1f}% ({keyword_success}개)")
        print(f"  - Word2Vec 유사도 성공률: {word2vec_rate:.1f}% ({word2vec_success}개)")
        print(f"  - 분류 실패율: {failure_rate:.1f}% ({failed_classifications}개)")
        
        print(f"\n💡 성능 개선 제안:")
        if failure_rate > 10:
            print(f"  - 분류 실패율이 {failure_rate:.1f}%로 높습니다. 유사도 임계값({self.classifier.similarity_threshold}) 조정을 고려하세요.")
        if word2vec_rate < 5:
            print("  - Word2Vec 기반 분류가 적습니다. 모델 경로와 품질을 확인하세요.")
        if keyword_rate > 80:
            print("  - 키워드 매칭 의존도가 높습니다. 더 다양한 표현을 위한 키워드 확장을 고려하세요.")
        
        print(f"\n✨ 전체 성능: {accuracy:.1f}%")
        if accuracy >= 85:
            print("  🎉 우수한 성능입니다!")
        elif accuracy >= 75:
            print("  👍 양호한 성능입니다.")
        else:
            print("  ⚠️  성능 개선이 필요합니다.")

def main():
    """메인 실행 함수"""
    print("MessageClassifier2 시나리오 테스트 시작")
    print("유사도 임계값을 조정하려면 similarity_threshold 매개변수를 변경하세요.")
    print()
    
    # 테스트 실행 (임계값 0.3로 설정)
    tester = Classifier2ScenarioTester(similarity_threshold=0.3)
    
    print("테스트 출력 옵션:")
    print("- 요약만: show_details=False, show_test_sentences=False")
    print("- 샘플 5개: show_details=True, show_test_sentences=False")
    print("- 모든 문장: show_details=False, show_test_sentences=True")
    print("현재는 요약 결과만 출력합니다.\n")
    
    # 전체 테스트 실행
    # 모든 테스트 문장을 보려면: show_test_sentences=True로 변경
    tester.run_all_tests(show_details=False, show_test_sentences=True)
    
    print("\n" + "="*80)
    print("✨ MessageClassifier2 테스트 완료!")
    print("KoNLPy + Word2Vec 기반 분류기의 성능을 확인했습니다.")
    print("\n💡 모든 테스트 문장을 보려면:")
    print("   tester.run_all_tests(show_test_sentences=True) 로 실행하세요.")
    print("\n⚙️  임계값을 조정하려면:")
    print("   Classifier2ScenarioTester(similarity_threshold=0.3) 처럼 생성하세요.")

if __name__ == "__main__":
    main()