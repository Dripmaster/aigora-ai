# -*- coding: utf-8 -*-
"""
MessageClassifier2 - 간소화된 Sum 방식 유사도 CJ 인재상 분류기 (다중 인재상 지원)

특징:
1. Sum 방식 Word2Vec 유사도 계산 (핵심 기능 유지)
2. KoNLPy 형태소 분석 (핵심 기능 유지) 
3. 단순화된 키워드 매칭 (2단계)
4. 간소화된 코드 구조
5. 일정 점수 이상의 여러 인재상 동시 분류 지원
"""

import re
import logging
from pathlib import Path
from konlpy.tag import Okt
from gensim.models import KeyedVectors


class MessageClassifier2:
    """간소화된 Sum 방식 CJ 인재상 분류기 (다중 인재상 지원)"""
    
    def __init__(self, 
                 similarity_threshold=0.15,
                 multi_trait_threshold=0.9,    # 최고점의 80% 이상
                 min_trait_score=60,           # 최소 60점 이상  
                 max_traits=3,                # 최대 3개까지
                 confidence_threshold=0.25):    # 신뢰도 임계값
        """분류기 초기화"""
        self.similarity_threshold = similarity_threshold
        self.multi_trait_threshold = multi_trait_threshold
        self.min_trait_score = min_trait_score
        self.max_traits = max_traits
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # KoNLPy 초기화
        try:
            self.okt = Okt()
            self.morphs_enabled = True
        except Exception as e:
            self.morphs_enabled = False
            self.logger.warning(f"KoNLPy 초기화 실패: {e}")
        
        # Word2Vec 모델 로드
        self.word2vec_model = self._load_word2vec_model()
        
        # CJ 인재상 키워드 정의 (형태소 분석 stem=True 어간 추출 최적화)
        self.cj_keywords = {
            "정직": [
                "솔직", "사실", "진실", "정직", "투명",
                "신뢰", "믿음", "인정", "양심", "사과", 
                "고백", "진솔", "정확", "털어놓"
            ],
            "열정": [
                "열정", "적극", "최선", "열심", "도전", 
                "의욕", "헌신", "활기", "열의", "몰입", 
                "끈기", "용기", "투지", "열심", "좋아하", 
                "빠지", "최고", "최선", "전력", "의지", "활력"
            ],
            "창의": [
                "혁신", "아이디어", "독창", "창의", "창조",
                "기발", "참신", "독특", "변화", "발견", 
                "실험", "설계", "구상", "색다르", "특별", 
                "신선", "특이", "재미있", "흥미롭", "제안", 
                "떠올리", "상상력"
            ],
            "존중": [
                "배려", "존중", "경청", "공감", "이해",
                "친절", "포용", "겸손", "소통", "팀워크", 
                "도움", "지원", "협조", "협력", "화합", 
                "매너", "함께", "서로", "같이", "입장", 
                "대화", "상대", "생각하", "느끼", "고려하", 
                "위하", "도와", "예의"
            ]
        }
        
        self.logger.info(f"MessageClassifier2 초기화 완료")
        self.logger.info(f"형태소 분석: {'활성화' if self.morphs_enabled else '비활성화'}")
        self.logger.info(f"Word2Vec: {'로드됨' if self.word2vec_model else '로드 실패'}") 
        self.logger.info(f"다중 분류 설정: 임계값={multi_trait_threshold}, 최소점수={min_trait_score}, 최대개수={max_traits}")
        self.logger.info(f"신뢰도 임계값: {confidence_threshold}")
    
    def _load_word2vec_model(self):
        """Word2Vec 모델 로드 (간소화)"""
        model_paths = [
            Path("/Users/cheon/Desktop/CJ/models/ko.bin"),
            Path("models/ko.bin"),
            Path("ko.bin")
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                try:
                    # 가장 일반적인 방식으로 로드
                    import pickle
                    with open(str(model_path), 'rb') as f:
                        model = pickle.load(f, encoding='latin-1')
                    
                    if hasattr(model, 'wv'):
                        self.logger.info(f"Word2Vec 모델 로드 성공: {model_path}")
                        return model.wv
                    elif hasattr(model, 'syn0') and hasattr(model, 'index2word'):
                        # 구버전 래퍼 생성
                        kv = KeyedVectors(vector_size=model.syn0.shape[1])
                        kv.key_to_index = {word: i for i, word in enumerate(model.index2word)}
                        kv.index_to_key = model.index2word[:]
                        kv.vectors = model.syn0.copy()
                        self.logger.info(f"구버전 Word2Vec 래퍼 생성 성공: {model_path}")
                        return kv
                except Exception as e:
                    self.logger.warning(f"모델 로드 실패 {model_path}: {e}")
                    continue
        
        self.logger.warning("Word2Vec 모델을 찾을 수 없습니다")
        return None
    
    def _analyze_morphemes(self, text):
        """형태소 분석 (간소화)"""
        if self.morphs_enabled:
            try:
                pos_tags = self.okt.pos(text, stem=True)
                morphemes = []
                for word, pos in pos_tags:
                    if pos in ['Noun', 'Adjective', 'Verb', 'Adverb'] and len(word) >= 2:
                        if re.match(r'^[가-힣]+$', word):
                            morphemes.append(word)
                return list(set(morphemes))
            except Exception as e:
                self.logger.error(f"형태소 분석 오류: {e}")
        
        # 폴백: 한글 단어 추출
        return list(set(re.findall(r'[가-힣]{2,}', text)))
    
    def _calculate_keyword_scores(self, morphemes, text):
        """키워드 매칭 (2단계 단순화)"""
        scores = {"정직": 0, "열정": 0, "창의": 0, "존중": 0}
        
        for trait, keywords in self.cj_keywords.items():
            for keyword in keywords:
                # 1. 정확 매칭 (20점)
                if keyword in morphemes:
                    scores[trait] += 20
                # 2. 원문 매칭 (10점)
                elif keyword in text:
                    scores[trait] += 10
        
        
        return scores
    
    def _calculate_sum_similarity(self, morphemes, trait):
        """Sum 방식 유사도 계산 (핵심 기능 유지)"""
        if not self.word2vec_model:
            return 0.0
        
        total_similarity = 0.0
        try:
            model_vocab = self.word2vec_model
            
            for morpheme in morphemes:
                if morpheme in model_vocab:
                    max_sim = 0.0
                    for keyword in self.cj_keywords[trait]:
                        if keyword in model_vocab:
                            sim = model_vocab.similarity(morpheme, keyword)
                            max_sim = max(max_sim, sim)
                    total_similarity += max_sim
            
            return total_similarity
        except Exception as e:
            self.logger.error(f"유사도 계산 오류: {e}")
            return 0.0
    
    def _normalize_scores(self, scores):
        """점수 정규화 (단순화)"""
        max_score = max(scores.values()) if any(scores.values()) else 0
        if max_score == 0:
            return {"정직": 0, "열정": 0, "창의": 0, "존중": 0}
        
        normalized = {}
        for trait, score in scores.items():
            normalized[trait] = min(100, int((score / max_score) * 100))
            if normalized[trait] > 0:
                normalized[trait] = max(5, normalized[trait])
        
        return normalized
    
    def _select_multiple_traits(self, normalized_scores):
        """다중 인재상 선별 로직 - 상위 2개 인재상 점수 차이가 20점 이하면 둘 다 출력"""
        # 점수 순으로 정렬
        sorted_traits = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_traits or sorted_traits[0][1] == 0:
            return [], {}
        
        selected_traits = []
        trait_details = {}
        
        # 1위는 항상 포함
        first_trait, first_score = sorted_traits[0]
        selected_traits.append(first_trait)
        trait_details[first_trait] = {"score": first_score, "confidence": first_score / 100.0}
        
        # 2위가 있고 1위와 점수 차이가 20점 이하면 포함
        if len(sorted_traits) > 1:
            second_trait, second_score = sorted_traits[1]
            if second_score > 0 and (first_score - second_score) <= 20:
                selected_traits.append(second_trait)
                trait_details[second_trait] = {"score": second_score, "confidence": second_score / 100.0}
        
        return selected_traits, trait_details
    
    def classify(self, text, user_id=""):
        """메시지 분류 메인 함수 - 다중 인재상 지원"""
        # 빈 메시지 및 짧은 메시지 처리 (CJ 인재상 키워드 예외)
        if not text or len(text.strip()) < 5:
            # CJ 인재상 키워드가 포함된 경우는 예외 처리
            cj_traits = ["정직", "열정", "창의", "존중"]
            has_cj_trait = any(trait in text for trait in cj_traits)
            
            if not has_cj_trait:
                return {
                    "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                    "primary_trait": "무응답",
                    "multiple_traits": [],
                    "summary": "메시지가 너무 짧음 (5글자 미만)",
                    "method": "skip",
                    "confidence": 0.0,
                    "morphemes": []
                }
        
        # 명시적 인재상 키워드 검출
        text_lower = text.lower()
        explicit_traits = []
        for trait in ["정직", "열정", "창의", "존중"]:
            if trait in text_lower:
                explicit_traits.append(trait)
        
        if explicit_traits:
            morphemes = self._analyze_morphemes(text)
            return {
                "cj_values": {t: 100 if t in explicit_traits else 0 for t in ["정직", "열정", "창의", "존중"]},
                "primary_trait": explicit_traits[0],
                "multiple_traits": explicit_traits,
                "summary": f"명시적 키워드 검출: {', '.join(explicit_traits)}",
                "method": "explicit_trait",
                "confidence": 1.0,
                "morphemes": morphemes
            }
        
        # 형태소 분석
        morphemes = self._analyze_morphemes(text)
        if not morphemes:
            return {
                "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                "primary_trait": "분석실패",
                "multiple_traits": [],
                "summary": "형태소 분석 실패",
                "method": "analysis_failed",
                "confidence": 0.0,
                "morphemes": []
            }
        
        # 키워드 매칭 시도
        keyword_scores = self._calculate_keyword_scores(morphemes, text)
        if sum(keyword_scores.values()) > 0:
            normalized_scores = self._normalize_scores(keyword_scores)
            selected_traits, trait_details = self._select_multiple_traits(normalized_scores)
            
            if selected_traits:
                primary_trait = selected_traits[0]
                confidence = normalized_scores[primary_trait] / 100.0
                
                # 신뢰도 임계값 검사
                if confidence < self.confidence_threshold:
                    return {
                        "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                        "primary_trait": "분류하지않음",
                        "multiple_traits": [],
                        "summary": f"신뢰도 부족으로 분류 실패 (신뢰도: {confidence:.3f})",
                        "method": "low_confidence",
                        "confidence": confidence,
                        "morphemes": morphemes
                    }
                
                return {
                    "cj_values": normalized_scores,
                    "primary_trait": primary_trait,
                    "multiple_traits": selected_traits,
                    "summary": f"키워드 매칭으로 '{primary_trait}' 분류",
                    "method": "keyword_matching", 
                    "confidence": confidence,
                    "morphemes": morphemes
                }
        
        # Sum 유사도 분석
        if self.word2vec_model:
            similarity_scores = {}
            for trait in ["정직", "열정", "창의", "존중"]:
                similarity_scores[trait] = self._calculate_sum_similarity(morphemes, trait)
            
            max_similarity = max(similarity_scores.values())
            threshold = self.similarity_threshold * len(morphemes)
            
            if max_similarity >= threshold:
                normalized_scores = self._normalize_scores(similarity_scores)
                selected_traits, trait_details = self._select_multiple_traits(normalized_scores)
                
                if selected_traits:
                    primary_trait = selected_traits[0]
                    confidence = min(1.0, max_similarity / len(morphemes))
                    
                    # 신뢰도 임계값 검사
                    if confidence < self.confidence_threshold:
                        return {
                            "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                            "primary_trait": "분류하지않음",
                            "multiple_traits": [],
                            "summary": f"신뢰도 부족으로 분류 실패 (신뢰도: {confidence:.3f})",
                            "method": "low_confidence",
                            "confidence": confidence,
                            "morphemes": morphemes
                        }
                    
                    return {
                        "cj_values": normalized_scores,
                        "primary_trait": primary_trait,
                        "multiple_traits": selected_traits,
                        "summary": f"유사도 분석으로 '{primary_trait}' 분류 (합계: {max_similarity:.3f})",
                        "method": "sum_similarity", 
                        "confidence": confidence,
                        "morphemes": morphemes
                    }
        
        # 분류 실패
        return {
            "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
            "primary_trait": "분류실패",
            "multiple_traits": [],
            "summary": "모든 분류 방법 임계값 미달",
            "method": "classification_failed",
            "confidence": 0.0,
            "morphemes": morphemes
        }
    
    def get_user_profile(self, user_id, messages):
        """사용자 프로필 생성"""
        if not messages:
            return {"error": "분석할 메시지가 없습니다"}
        
        classifications = []
        method_counts = {
            "explicit_trait": 0, "keyword_matching": 0, "sum_similarity": 0,
            "classification_failed": 0, "skip": 0, "analysis_failed": 0, "low_confidence": 0
        }
        
        for msg in messages:
            result = self.classify(msg["text"], user_id)
            classifications.append(result)
            method = result.get("method", "unknown")
            if method in method_counts:
                method_counts[method] += 1
        
        # 평균 점수 계산
        avg_cj_values = {}
        for trait in ["정직", "열정", "창의", "존중"]:
            scores = [c["cj_values"][trait] for c in classifications if c["cj_values"][trait] > 0]
            avg_cj_values[trait] = round(sum(scores) / len(scores)) if scores else 0
        
        # 상위 특성
        top_traits = sorted(avg_cj_values.items(), key=lambda x: x[1], reverse=True)[:2]
        
        # 성공률 계산
        total = len(messages)
        success = method_counts["explicit_trait"] + method_counts["keyword_matching"] + method_counts["sum_similarity"]
        success_rate = success / total if total > 0 else 0
        
        return {
            "user_id": user_id,
            "message_count": total,
            "avg_cj_values": avg_cj_values,
            "top_traits": [trait for trait, _ in top_traits],
            "classification_methods": method_counts,
            "success_rate": round(success_rate * 100, 1),
            "overall_summary": f"{user_id}님의 주요 특성: {', '.join([trait for trait, _ in top_traits[:2]])}"
        }
    
    def get_classification_stats(self):
        """분류기 설정 정보 반환 (간소화)"""
        return {
            "classifier_type": "MessageClassifier2",
            "similarity_threshold": self.similarity_threshold,
            "multi_trait_threshold": self.multi_trait_threshold,
            "min_trait_score": self.min_trait_score,
            "max_traits": self.max_traits,
            "confidence_threshold": self.confidence_threshold,
            "morphs_enabled": self.morphs_enabled,
            "word2vec_loaded": self.word2vec_model is not None
        }


