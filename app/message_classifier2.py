# -*- coding: utf-8 -*-
"""
MessageClassifier4 - 간소화된 Sum 방식 유사도 CJ 인재상 분류기

특징:
1. Sum 방식 Word2Vec 유사도 계산 (핵심 기능 유지)
2. KoNLPy 형태소 분석 (핵심 기능 유지) 
3. 단순화된 키워드 매칭 (2단계)
4. 간소화된 코드 구조
"""

import re
import logging
from pathlib import Path
from konlpy.tag import Okt
from gensim.models import KeyedVectors


class MessageClassifier4:
    """간소화된 Sum 방식 CJ 인재상 분류기 (Version 4)"""
    
    def __init__(self, similarity_threshold=0.1):
        """분류기 초기화"""
        self.similarity_threshold = similarity_threshold
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
                "신뢰", "믿음", "인정", "양심", "사과", "고백", "진솔", "참",
                "믿", "정확", "털어놓" ,"진심"
            ],
            "열정": [
                "열정", "적극", "최선", "열심", "도전", "의욕", "헌신", "활기", "열의",
                "몰입", "끈기", "용기", "투지", "노력",
                "좋아하", "빠지", "최고", "최선", "전력", "의지"
            ],
            "창의": [
                "혁신", "아이디어", "독창", "창의", "창조", "기발", "참신", "독특",
                "변화", "발견", "실험", "설계", "구상",
                "새롭다", "색다르", "특별", "신선", "특이", "재미있", "흥미롭",
                "제안", "떠올리", "만들", "상상력"
            ],
            "존중": [
                "배려", "존중", "경청", "공감", "이해", "친절", "포용", "겸손",
                "소통", "팀워크", "도움", "지원", "협조", "협력", "화합", "매너",
                "함께", "서로", "같이", "입장", "대화", "상대",
                "생각하", "느끼", "고려하", "위하", "도와","예의"
            ]
        }
        
        self.logger.info(f"MessageClassifier4 초기화 완료")
        self.logger.info(f"형태소 분석: {'활성화' if self.morphs_enabled else '비활성화'}")
        self.logger.info(f"Word2Vec: {'로드됨' if self.word2vec_model else '로드 실패'}")
    
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
    
    def classify(self, text, user_id=""):
        """메시지 분류 메인 함수"""
        # 빈 메시지 처리
        if not text or len(text.strip()) < 2:
            return {
                "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                "primary_trait": "무응답",
                "summary": "메시지가 너무 짧음",
                "method": "skip",
                "confidence": 0.0,
                "morphemes": []
            }
        
        # 명시적 인재상 키워드 검출
        text_lower = text.lower()
        for trait in ["정직", "열정", "창의", "존중"]:
            if trait in text_lower:
                morphemes = self._analyze_morphemes(text)
                return {
                    "cj_values": {t: 100 if t == trait else 0 for t in ["정직", "열정", "창의", "존중"]},
                    "primary_trait": trait,
                    "summary": f"명시적 '{trait}' 키워드 검출",
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
                "summary": "형태소 분석 실패",
                "method": "analysis_failed",
                "confidence": 0.0,
                "morphemes": []
            }
        
        # 키워드 매칭 시도
        keyword_scores = self._calculate_keyword_scores(morphemes, text)
        if sum(keyword_scores.values()) > 0:
            normalized_scores = self._normalize_scores(keyword_scores)
            primary_trait = max(normalized_scores, key=normalized_scores.get)
            confidence = normalized_scores[primary_trait] / 100.0
            
            return {
                "cj_values": normalized_scores,
                "primary_trait": primary_trait,
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
                best_trait = max(normalized_scores, key=normalized_scores.get)
                confidence = min(1.0, max_similarity / len(morphemes))
                
                return {
                    "cj_values": normalized_scores,
                    "primary_trait": best_trait,
                    "summary": f"유사도 분석으로 '{best_trait}' 분류 (합계: {max_similarity:.3f})",
                    "method": "sum_similarity", 
                    "confidence": confidence,
                    "morphemes": morphemes
                }
        
        # 분류 실패
        return {
            "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
            "primary_trait": "분류실패",
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
            "classification_failed": 0, "skip": 0, "analysis_failed": 0
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