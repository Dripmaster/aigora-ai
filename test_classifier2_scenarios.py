# -*- coding: utf-8 -*-
"""
MessageClassifier4 시나리오 테스트
기존 시나리오 파일들을 사용하여 간소화된 MessageClassifier4의 성능을 검증합니다.
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
from scenario5_discussion_cases import get_scenario5_discussion_cases


class Classifier2ScenarioTester:
    """MessageClassifier2로 모든 시나리오 테스트"""
    
    def __init__(self, similarity_threshold=0.1):
        self.classifier = MessageClassifier2(similarity_threshold)
        
        # 모델 초기화 상태 확인
        self._check_model_status()
        
        self.results = {
            "total_tests": 0,
            "correct_predictions": 0,
            "scenarios": {},
            "method_stats": {
                "explicit_trait": 0,
                "keyword_matching": 0,
                "sum_similarity": 0,
                "classification_failed": 0,
                "skip": 0,
                "analysis_failed": 0,
                "error": 0
            }
        }
    
    def _check_model_status(self):
        """모델 초기화 상태 확인 및 출력"""
        print("🔍 MessageClassifier2 모델 상태 확인")
        print("=" * 50)
        
        # KoNLPy 형태소 분석기 상태
        morphs_status = "✅ 활성화" if self.classifier.morphs_enabled else "❌ 비활성화"
        print(f"KoNLPy 형태소 분석기: {morphs_status}")
        
        if self.classifier.morphs_enabled:
            try:
                # 간단한 형태소 분석 테스트
                test_result = self.classifier.okt.pos("테스트 문장입니다")
                print(f"  → 형태소 분석 테스트: ✅ 정상 ({len(test_result)}개 형태소 추출)")
            except Exception as e:
                print(f"  → 형태소 분석 테스트: ❌ 실패 - {e}")
        
        # Word2Vec 모델 상태
        w2v_status = "✅ 로드됨" if self.classifier.word2vec_model else "❌ 로드 실패"
        print(f"Word2Vec 모델: {w2v_status}")
        
        if self.classifier.word2vec_model:
            try:
                # 모델 기본 정보
                if hasattr(self.classifier.word2vec_model, 'key_to_index'):
                    vocab_size = len(self.classifier.word2vec_model.key_to_index)
                    print(f"  → 어휘 크기: {vocab_size:,}개")
                    
                    # 테스트 단어로 유사도 계산 테스트
                    test_words = ["정직", "열정", "창의", "존중"]
                    available_words = [w for w in test_words if w in self.classifier.word2vec_model]
                    print(f"  → CJ 핵심 키워드 포함: {len(available_words)}/4개 ({', '.join(available_words)})")
                    
                    if len(available_words) >= 2:
                        try:
                            sim = self.classifier.word2vec_model.similarity(available_words[0], available_words[1])
                            print(f"  → 유사도 계산 테스트: ✅ {available_words[0]}-{available_words[1]} = {sim:.3f}")
                        except Exception as e:
                            print(f"  → 유사도 계산 테스트: ❌ 실패 - {e}")
                else:
                    print("  → 모델 어휘 정보를 확인할 수 없습니다")
            except Exception as e:
                print(f"  → 모델 상태 확인 실패: {e}")
        else:
            print("  → Word2Vec 모델이 없어 유사도 기반 분류가 작동하지 않습니다")
            print("  → 키워드 매칭과 명시적 특성 검출만 사용됩니다")
        
        # 분류기 설정 정보
        print(f"유사도 임계값: {self.classifier.similarity_threshold}")
        print(f"CJ 키워드 세트: 4개 특성 × {sum(len(keywords) for keywords in self.classifier.cj_keywords.values())}개 키워드")
        
        # 간단한 분류 테스트
        print("\n🧪 간단한 분류 테스트:")
        test_cases = [
            ("정직하게 말씀드리겠습니다", "정직"),
            ("열심히 노력하겠습니다", "열정"),
            ("새로운 아이디어를 제안합니다", "창의"),
            ("고객을 배려해야 합니다", "존중")
        ]
        
        for text, expected in test_cases:
            try:
                result = self.classifier.classify(text, "test")
                predicted = result["primary_trait"]
                method = result.get("method", "unknown")
                status = "✅" if predicted == expected else "❌"
                print(f"  {status} '{text}' → {predicted} ({method})")
            except Exception as e:
                print(f"  ❌ '{text}' → 분류 실패: {e}")
        
        print("=" * 50)
        print()
    
    def test_scenario(self, scenario_name: str, test_cases: list, show_details: bool = False, 
                      show_test_sentences: bool = False) -> dict:
        """특정 시나리오 테스트"""
        print(f"\n{'='*20} {scenario_name} 테스트 {'='*20}")
        
        scenario_results = {
            "total": len(test_cases),
            "correct": 0,
            "trait_stats": {},
            "method_stats": {"explicit_trait": 0, "keyword_matching": 0, "sum_similarity": 0, 
                           "classification_failed": 0, "skip": 0, "analysis_failed": 0, "error": 0},
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
                    
                    # 분류 실패나 오답인 경우 추가 정보 표시
                    if predicted != expected or method == "classification_failed":
                        print(f"     점수: {result['cj_values']}")
                        
                        # 분류 실패 특별 표시
                        if method == "classification_failed":
                            print(f"     ⚠️  분류 실패: 모든 분류 방법이 임계값 미달")
                            if result.get('summary'):
                                print(f"     실패 원인: {result['summary']}")
                    
                    if result.get('morphemes'):
                        print(f"     형태소: {result['morphemes']}")
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
                if result.get('morphemes'):
                    print(f"    형태소: {result['morphemes'][:5]}...")
                print()
        
        # 시나리오별 결과 출력
        accuracy = scenario_results["correct"] / scenario_results["total"] * 100
        print(f"\n{scenario_name} 정확도: {scenario_results['correct']}/{scenario_results['total']} = {accuracy:.1f}%")
        
        # 방법별 통계
        print(f"분류 방법별 통계:")
        for method, count in scenario_results["method_stats"].items():
            if count > 0:
                percentage = count / scenario_results["total"] * 100
                status_icon = "⚠️ " if method == "classification_failed" else ""
                print(f"  {status_icon}{method}: {count}개 ({percentage:.1f}%)")
        
        # 인재상별 정확도
        print(f"인재상별 정확도:")
        for trait, stats in scenario_results["trait_stats"].items():
            trait_accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {trait}: {stats['correct']}/{stats['total']} = {trait_accuracy:.1f}%")
        
        return scenario_results
    
    def show_error_analysis(self, scenario_name: str, errors: list, max_errors: int = None):
        """오분류 사례 분석 - 모든 오답 표시"""
        if not errors:
            print(f"\n{scenario_name}: 모든 테스트 케이스가 정확히 분류되었습니다! 🎉")
            return
        
        # max_errors가 None이면 모든 오답을 표시
        if max_errors is None:
            max_errors = len(errors)
            print(f"\n{scenario_name} 오분류 사례 분석 (총 {len(errors)}개 전체):")
        else:
            print(f"\n{scenario_name} 오분류 사례 분석 (총 {len(errors)}개 중 {min(max_errors, len(errors))}개):")
        
        print("-" * 80)
        
        for i, error in enumerate(errors[:max_errors]):
            print(f"{i+1}. 문장: '{error['text']}'")
            print(f"   예상: {error['expected']} vs 예측: {error['predicted']}")
            print(f"   방법: {error['method']}, 신뢰도: {error['confidence']:.3f}")
            print(f"   점수: {error['scores']}")
            
            # 분류 실패 특별 표시
            if error['method'] == 'classification_failed':
                print(f"   ⚠️  분류 실패: 모든 분류 방법이 임계값 미달")
            else:
                # 점수 차이 분석 (분류 실패가 아닌 경우만)
                expected_score = error['scores'][error['expected']]
                predicted_score = error['scores'][error['predicted']]
                score_diff = predicted_score - expected_score
                print(f"   점수 차이: {error['predicted']}({predicted_score}) - {error['expected']}({expected_score}) = {score_diff:+d}")
            print()
    
    def run_all_tests(self, show_details: bool = False, show_test_sentences: bool = False):
        """모든 시나리오 테스트 실행"""
        print("🧪 MessageClassifier2 - 모든 시나리오 종합 테스트")
        print("=" * 80)
        print("간소화된 Sum 방식 분류기로 5개 시나리오의 실제 토론 발언 분류 성능 검증")
        if show_test_sentences:
            print("📝 모든 테스트 문장과 결과를 출력합니다.")
        print()
        
        # 각 시나리오 테스트
        scenarios = [
            ("시나리오 1: 갓 구운 빵의 비밀", get_scenario1_discussion_cases()),
            ("시나리오 2: 알바생 혜경이의 열정", get_scenario2_discussion_cases()),
            ("시나리오 3: 신메뉴 개발 아이디어", get_scenario3_discussion_cases()),
            ("시나리오 4: 고객 불만 응대", get_scenario4_discussion_cases()),
            ("시나리오 5: 빕스 조리 근무자", get_scenario5_discussion_cases())
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
                status_icon = "⚠️ " if method == "classification_failed" else ""
                print(f"  {status_icon}{method}: {count}개 ({percentage:.1f}%)")
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
            self.show_error_analysis(scenario_name, results["errors"], max_errors=None)  # 모든 오답 표시
        
        # 성능 비교 및 분석
        self.show_performance_analysis(overall_accuracy, total_tests)
        
        # 오답 문장들을 파일로 저장
        self.save_incorrect_sentences()
    
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
        
        explicit_success = self.results["method_stats"]["explicit_trait"]
        keyword_success = self.results["method_stats"]["keyword_matching"]
        similarity_success = self.results["method_stats"]["sum_similarity"] 
        failed_classifications = self.results["method_stats"]["classification_failed"]
        
        explicit_rate = explicit_success / total_tests * 100
        keyword_rate = keyword_success / total_tests * 100
        similarity_rate = similarity_success / total_tests * 100
        failure_rate = failed_classifications / total_tests * 100
        
        print(f"📈 분류 성공률 분석:")
        print(f"  - 명시적 인재상 키워드: {explicit_rate:.1f}% ({explicit_success}개)")
        print(f"  - 키워드 매칭 성공률: {keyword_rate:.1f}% ({keyword_success}개)")
        print(f"  - Sum 유사도 성공률: {similarity_rate:.1f}% ({similarity_success}개)")
        print(f"  - ⚠️  분류 실패율: {failure_rate:.1f}% ({failed_classifications}개)")
        
        print(f"\n💡 성능 개선 제안:")
        if failure_rate > 15:
            print(f"  - ⚠️  분류 실패율이 {failure_rate:.1f}%로 높습니다!")
            print(f"    → 유사도 임계값({self.classifier.similarity_threshold}) 조정을 고려하세요.")
            print(f"    → Word2Vec 모델 로드 상태를 확인하세요.")
        if failure_rate > 0:
            print(f"  - 분류 실패 케이스 {failed_classifications}개가 발견되었습니다.")
        if similarity_rate < 10:
            print("  - Sum 유사도 기반 분류가 적습니다. Word2Vec 모델 경로와 품질을 확인하세요.")
        if keyword_rate > 80:
            print("  - 키워드 매칭 의존도가 높습니다. 더 다양한 표현을 위한 키워드 확장을 고려하세요.")
        if explicit_rate > 30:
            print("  - 명시적 키워드 의존도가 높습니다. 실제 상황에서는 명시적 표현이 적을 수 있습니다.")
        
        print(f"\n✨ 전체 성능: {accuracy:.1f}%")
        if accuracy >= 85:
            print("  🎉 우수한 성능입니다!")
        elif accuracy >= 75:
            print("  👍 양호한 성능입니다.")
        elif accuracy >= 65:
            print("  ⚠️  보통 성능입니다. 개선을 고려하세요.")
        else:
            print("  🔧 성능 개선이 필요합니다.")
    
    def save_incorrect_sentences(self):
        """오답 문장들을 텍스트 파일로 저장"""
        from datetime import datetime
        
        # 현재 시각으로 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"incorrect_sentences_{timestamp}.txt"
        
        # 모든 시나리오의 오답 문장들 수집
        all_errors = []
        total_errors = 0
        
        print("📋 오답 수집 상황:")
        for scenario_name, results in self.results["scenarios"].items():
            scenario_errors = results["errors"]
            print(f"  - {scenario_name}: {len(scenario_errors)}개 오답")
            total_errors += len(scenario_errors)
            
            for error in scenario_errors:
                all_errors.append({
                    "scenario": scenario_name,
                    "text": error["text"],
                    "expected": error["expected"],
                    "predicted": error["predicted"],
                    "method": error["method"],
                    "confidence": error["confidence"],
                    "scores": error["scores"]
                })
        
        print(f"📊 총 수집된 오답: {len(all_errors)}개 (예상: {total_errors}개)")
        
        if total_errors == 0:
            print("🎉 모든 문장이 정확히 분류되어 오답 파일을 생성하지 않습니다!")
            return
        
        # 파일에 오답 문장들 저장
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# MessageClassifier2 오답 문장 분석 결과\n")
                f.write(f"# 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 총 오답 개수: {total_errors}개\n")
                f.write("=" * 80 + "\n\n")
                
                # 시나리오별로 그룹화하여 저장
                current_scenario = ""
                error_count = 1
                
                try:
                    for error in all_errors:
                        # 새로운 시나리오 시작
                        if error["scenario"] != current_scenario:
                            current_scenario = error["scenario"]
                            f.write(f"\n## {current_scenario}\n")
                            f.write("-" * 50 + "\n\n")
                            f.flush()  # 버퍼 강제 쓰기
                        
                        # 오답 정보 저장
                        f.write(f"{error_count}. 문장: {error['text']}\n")
                        f.write(f"   예상: {error['expected']} → 결과: {error['predicted']}\n")
                        f.write(f"   분류방법: {error['method']}\n")
                        f.write(f"   신뢰도: {error['confidence']:.3f}\n")
                        f.write(f"   점수: {error['scores']}\n")
                        
                        # 분류 실패 특별 표시
                        if error['method'] == 'classification_failed':
                            f.write(f"   ⚠️  분류 실패: 모든 분류 방법이 임계값 미달\n")
                        else:
                            # 점수 차이 계산 (분류 실패가 아닌 경우만)
                            try:
                                expected_score = error['scores'][error['expected']]
                                predicted_score = error['scores'][error['predicted']]
                                score_diff = predicted_score - expected_score
                                f.write(f"   점수차이: {error['predicted']}({predicted_score}) - {error['expected']}({expected_score}) = {score_diff:+d}\n")
                            except KeyError as e:
                                f.write(f"   점수차이: 계산 오류 - {e}\n")
                        
                        f.write("\n")
                        f.flush()  # 각 오답마다 버퍼 강제 쓰기
                        error_count += 1
                        
                except Exception as write_error:
                    f.write(f"\n❌ 오답 저장 중 오류 발생 (#{error_count}): {write_error}\n")
                    f.write(f"남은 오답 수: {len(all_errors) - error_count + 1}개\n")
                    f.flush()
                
                # 통계 정보 추가
                f.write("\n" + "=" * 80 + "\n")
                f.write("## 오답 통계 분석\n")
                f.write("=" * 80 + "\n\n")
                
                # 시나리오별 오답 수
                f.write("### 시나리오별 오답 개수:\n")
                for scenario_name, results in self.results["scenarios"].items():
                    error_count = len(results["errors"])
                    total_count = results["total"]
                    error_rate = error_count / total_count * 100 if total_count > 0 else 0
                    f.write(f"- {scenario_name}: {error_count}개 (전체 {total_count}개 중 {error_rate:.1f}%)\n")
                
                f.write("\n")
                
                # 분류 방법별 오답 통계
                method_errors = {}
                for error in all_errors:
                    method = error["method"]
                    method_errors[method] = method_errors.get(method, 0) + 1
                
                f.write("### 분류 방법별 오답 개수:\n")
                for method, count in method_errors.items():
                    percentage = count / total_errors * 100
                    f.write(f"- {method}: {count}개 ({percentage:.1f}%)\n")
                
                f.write("\n")
                
                # 인재상별 오답 통계  
                trait_errors = {}
                for error in all_errors:
                    expected = error["expected"]
                    predicted = error["predicted"]
                    key = f"{expected} → {predicted}"
                    trait_errors[key] = trait_errors.get(key, 0) + 1
                
                f.write("### 인재상별 오답 패턴 (상위 10개):\n")
                sorted_trait_errors = sorted(trait_errors.items(), key=lambda x: x[1], reverse=True)
                for pattern, count in sorted_trait_errors[:10]:
                    percentage = count / total_errors * 100
                    f.write(f"- {pattern}: {count}개 ({percentage:.1f}%)\n")
            
            print(f"\n📄 오답 문장 분석 결과가 '{filename}' 파일에 저장되었습니다.")
            print(f"   총 {total_errors}개의 오답 문장과 상세 분석이 포함되어 있습니다.")
            print(f"   실제 저장된 오답: {error_count-1}개")
            
            if error_count-1 != total_errors:
                print(f"⚠️  경고: 예상 오답 수({total_errors})와 저장된 오답 수({error_count-1})가 다릅니다!")
            
        except Exception as e:
            print(f"❌ 오답 파일 저장 중 오류 발생: {e}")
        
        


def main():
    """메인 실행 함수"""
    print("MessageClassifier2 시나리오 테스트 시작")
    print("간소화된 분류기의 성능을 검증합니다. (5개 시나리오 포함)")
    print()
    
    # 테스트 실행 (임계값 0.1로 설정 - 더 민감하게)
    tester = Classifier2ScenarioTester(similarity_threshold=0.1)
    
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
    print("간소화된 Sum 방식 분류기의 성능을 확인했습니다. (5개 시나리오 전체)")
    print("\n💡 모든 테스트 문장을 보려면:")
    print("   tester.run_all_tests(show_test_sentences=True) 로 실행하세요.")
    print("\n⚙️  임계값을 조정하려면:")
    print("   Classifier2ScenarioTester(similarity_threshold=0.1) 처럼 생성하세요.")
    print("   - 핵심 기능: 유지 (형태소 분석 + Sum 유사도)")
    print("   - 간소화: 키워드 매칭, Word2Vec 로드, 문맥 패턴")


if __name__ == "__main__":
    main()