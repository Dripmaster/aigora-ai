"""
MessageClassifier2 - KoNLPy와 Word2Vec을 활용한 CJ 인재상 분류기

특징:
1. KoNLPy Okt를 사용한 형태소 분석
2. 형태소 기반 정확한 키워드 매칭
3. Word2Vec 유사도 기반 분류 (키워드 매칭 실패시)
4. 임계값 기반 분류 시스템
"""

import re
import logging
from typing import Dict, List
from pathlib import Path

from konlpy.tag import Okt
from gensim.models import KeyedVectors


class MessageClassifier2:
    """KoNLPy와 Word2Vec을 활용한 CJ 인재상 분류기 (Version 2)"""
    
    def __init__(self, similarity_threshold=0.3):
        """분류기 초기화"""
        self.similarity_threshold = similarity_threshold
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # KoNLPy 초기화
        try:
            self.okt = Okt()
            self.morphs_enabled = True
            self.logger.info("KoNLPy Okt 형태소 분석기 초기화 완료")
        except Exception as e:
            self.morphs_enabled = False
            self.logger.warning(f"KoNLPy 초기화 실패: {e}")
        
        # Word2Vec 모델 경로 찾기
        self.model_path = self._find_model_path()
        
        # Word2Vec 모델 로드
        self.word2vec_model = None
        self._load_word2vec_model()
        
        # CJ 인재상 형태소 기반 키워드 정의 (최적화된 키워드 세트)
        self.cj_keywords = {
            "정직": {
                "핵심": ["솔직", "사실", "진실", "정직", "투명", "현실", "실제"],
                "복합": ["솔직히", "사실대로", "현실적", "실제로", "털어놓", "고백"],
                "일반": ["확실", "분명", "성실", "신뢰", "믿음", "인정", "명확", "정확"],
                "보조": ["참", "진정", "양심", "떳떳", "사과", "바르", "올바르"]
            },
            "열정": {
                "핵심": ["열정", "적극", "최선", "열심", "도전", "의욕", "헌신", "활기", "열의", "투지", "빠져", "몰두"],
                "복합": ["열정적", "적극적", "열심히", "최선을다해", "전력", "의욕적", "헌신적", "도전적", "푹빠져", "몰두하"],
                "일반": ["노력", "성장", "발전", "집중", "몰입", "진심", "실행", "추진", "끈기"],
                "보조": ["파이팅", "화이팅", "좋", "훌륭", "대단", "멋있", "용기", "활발"]
            },
            "창의": {
                "핵심": ["혁신", "아이디어", "독창", "창의", "창조", "기발", "참신", "독특", "발명"],
                "복합": ["혁신적", "창의적", "독창적", "참신하", "기발하", "새로운", "색다른"],
                "일반": ["개선", "변화", "개발", "발견", "실험", "기획", "설계", "구상"],
                "보조": ["새로", "특별", "다른", "특이", "신선", "재미", "흥미"]
            },
            "존중": {
                "핵심": ["배려", "존중", "경청", "공감", "이해", "친절", "포용", "겸손", "사려"],
                "복합": ["배려하", "존중하", "이해하", "공감하", "고객입장", "함께", "협력"],
                "일반": ["소통", "팀워크", "도움", "지원", "협조", "화합", "매너", "예의"],
                "보조": ["같이", "서로", "마음", "입장", "고객", "대화", "상대"]
            }
        }
        
        # 점수 가중치
        self.score_weights = {
            "핵심": 35,
            "복합": 28,
            "일반": 18,
            "보조": 8
        }
        
        self.logger.info("MessageClassifier2 초기화 완료")
        self.logger.info(f"   - 형태소 분석: {'활성화' if self.morphs_enabled else '비활성화'}")
        self.logger.info(f"   - Word2Vec 모델: {'로드됨' if self.word2vec_model else '로드 실패'}")
        self.logger.info(f"   - 유사도 임계값: {similarity_threshold}")
    
    def _find_model_path(self):
        """Word2Vec 모델 파일 경로 찾기"""
        # 절대 경로로 직접 지정
        model_path = Path("/Users/cheon/Desktop/CJ/models/ko.bin")
        if model_path.exists():
            return model_path
        
        # 백업 경로들
        possible_paths = [
            Path("models/ko.bin"),
            Path("models/ko.vec"),
            Path("ko/ko.bin"),
            Path("ko/ko.vec"),
            Path("ko.bin"),
            Path("ko.vec")
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    def _load_word2vec_model(self):
        """Word2Vec 모델 로드 (gensim 4.x 호환성 개선)"""
        try:
            if not self.model_path:
                self.logger.warning("Word2Vec 모델 파일을 찾을 수 없습니다")
                self.logger.info("models/ 폴더에 ko.bin 파일을 준비해주세요")
                return
            
            self.logger.info(f"Word2Vec 모델 로드 중: {self.model_path}")
            
            # 여러 방식으로 모델 로드 시도
            model_loaded = False
            
            if str(self.model_path).endswith('.bin'):
                # 방법 1: 구버전 gensim Word2Vec 모델 로드 시도 (latin-1 encoding)
                try:
                    import warnings
                    import pickle
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        # latin-1 encoding으로 구버전 모델 로드
                        with open(str(self.model_path), 'rb') as f:
                            model = pickle.load(f, encoding='latin-1')
                        
                        self.logger.info(f"구버전 모델 로드 성공: {type(model)}")
                        
                        # 구버전 gensim 모델 속성 확인 및 적응
                        if hasattr(model, 'wv'):
                            # gensim 1.x+ 형식
                            self.word2vec_model = model.wv
                            self.logger.info("Word2Vec.wv 모델 로드 완료")
                            model_loaded = True
                        elif hasattr(model, 'syn0') and hasattr(model, 'index2word'):
                            # gensim 0.x 형식 - KeyedVectors 래퍼 생성
                            self.word2vec_model = self._create_keyed_vectors_wrapper(model)
                            self.logger.info("구버전 Word2Vec 모델 래퍼 생성 완료")
                            model_loaded = True
                        else:
                            self.logger.warning("모델 구조를 인식할 수 없음")
                        
                except Exception as e1:
                    self.logger.warning(f"latin-1 pickle 로드 실패: {e1}")
                    
                    # 방법 2: 표준 gensim 로드 시도  
                    try:
                        from gensim.models import Word2Vec
                        model = Word2Vec.load(str(self.model_path))
                        
                        if hasattr(model, 'wv'):
                            self.word2vec_model = model.wv
                            self.logger.info("표준 Word2Vec.wv 모델 로드 완료")
                            model_loaded = True
                        else:
                            self.word2vec_model = model
                            self.logger.info("표준 Word2Vec 모델 로드 완료")
                            model_loaded = True
                            
                    except Exception as e2:
                        self.logger.warning(f"표준 Word2Vec.load 실패: {e2}")
                        
                        # 방법 3: KeyedVectors로 직접 로드 시도
                        try:
                            self.word2vec_model = KeyedVectors.load_word2vec_format(
                                str(self.model_path), 
                                binary=True
                            )
                            self.logger.info("KeyedVectors 바이너리 모델 로드 완료")
                            model_loaded = True
                        except Exception as e3:
                            self.logger.warning(f"KeyedVectors 바이너리 로드 실패: {e3}")
                        
            elif str(self.model_path).endswith('.vec'):
                # 텍스트 형식 모델 로드
                try:
                    self.word2vec_model = KeyedVectors.load_word2vec_format(
                        str(self.model_path), 
                        binary=False
                    )
                    self.logger.info("KeyedVectors 텍스트 모델 로드 완료")
                    model_loaded = True
                except Exception as e:
                    self.logger.warning(f"KeyedVectors 텍스트 로드 실패: {e}")
            
            if not model_loaded:
                self.logger.error("모든 Word2Vec 모델 로드 방식 실패")
                self.word2vec_model = None
                
        except Exception as e:
            self.logger.error(f"Word2Vec 모델 로드 중 예상치 못한 오류: {e}")
            self.word2vec_model = None
    
    def _create_keyed_vectors_wrapper(self, old_model):
        """구버전 gensim 모델을 현재 인터페이스로 래핑"""
        try:
            from gensim.models import KeyedVectors
            import numpy as np
            
            # 새로운 KeyedVectors 객체 생성
            kv = KeyedVectors(vector_size=old_model.syn0.shape[1])
            
            # 구버전 모델에서 데이터 복사
            if hasattr(old_model, 'index2word') and hasattr(old_model, 'syn0'):
                # 어휘와 벡터 복사
                vocab_size = len(old_model.index2word)
                kv.key_to_index = {word: i for i, word in enumerate(old_model.index2word)}
                kv.index_to_key = old_model.index2word[:]
                kv.vectors = old_model.syn0.copy()
                
                self.logger.info(f"래퍼 생성 완료: 어휘 크기 {vocab_size}, 벡터 차원 {old_model.syn0.shape[1]}")
                return kv
            else:
                self.logger.error("구버전 모델에서 필요한 속성을 찾을 수 없음")
                return None
                
        except Exception as e:
            self.logger.error(f"KeyedVectors 래퍼 생성 실패: {e}")
            return None
    
    def _analyze_morphemes(self, text):
        """텍스트를 형태소 단위로 분석 (정밀 분석)"""
        if self.morphs_enabled:
            try:
                # Okt를 사용한 품사 태깅
                pos_tags = self.okt.pos(text, stem=True)
                
                # 명사, 형용사, 동사, 부사만 추출
                meaningful_morphemes = []
                for word, pos in pos_tags:
                    # 명사(Noun), 형용사(Adjective), 동사(Verb), 부사(Adverb)
                    if pos in ['Noun', 'Adjective', 'Verb', 'Adverb'] and len(word) >= 2:
                        # 한글만 포함하는 단어
                        if re.match(r'^[가-힣]+$', word):
                            meaningful_morphemes.append(word)
                
                # 추가: 원본 텍스트에서 복합어 추출
                compound_words = re.findall(r'[가-힣]{3,}', text)
                meaningful_morphemes.extend(compound_words)
                
                # 중복 제거
                return list(set(meaningful_morphemes))
                
            except Exception as e:
                self.logger.error(f"형태소 분석 중 오류: {e}")
        
        # 폴백: 간단한 한글 단어 추출
        fallback_words = re.findall(r'[가-힣]{2,}', text)
        return list(set(fallback_words))
    
    def _calculate_all_keyword_scores(self, morphemes, original_text=None):
        """모든 특성에 대한 키워드 점수를 동시에 계산하여 중복 해결"""
        # 전체 키워드 매칭 결과 저장
        all_matches = {}
        trait_scores = {"정직": 0, "열정": 0, "창의": 0, "존중": 0}
        
        # 형태소 텍스트 생성
        morpheme_text = "".join(morphemes)
        if original_text is None:
            original_text = " ".join(morphemes)
        
        # 각 특성별로 매칭되는 키워드 찾기
        for trait in trait_scores.keys():
            all_matches[trait] = []
            keywords_dict = self.cj_keywords[trait]
            
            for level, keywords in keywords_dict.items():
                weight = self.score_weights[level]
                
                for keyword in keywords:
                    match_info = self._find_keyword_match(keyword, morphemes, morpheme_text, original_text)
                    if match_info:
                        all_matches[trait].append({
                            'keyword': keyword,
                            'level': level,
                            'weight': weight,
                            'match_type': match_info['match_type'],
                            'confidence': match_info['confidence']
                        })
        
        # 중복 키워드 해결 및 점수 계산
        used_keywords = set()
        
        # 1단계: 핵심 키워드 우선 처리
        for level in ["핵심", "복합", "일반", "보조"]:
            for trait in trait_scores.keys():
                for match in all_matches[trait]:
                    if match['level'] == level and match['keyword'] not in used_keywords:
                        # 키워드가 이미 다른 특성에서 사용되었는지 확인
                        conflict_traits = self._check_keyword_conflicts(match['keyword'], all_matches, used_keywords)
                        
                        if not conflict_traits:
                            # 충돌 없음 - 바로 점수 부여
                            trait_scores[trait] += match['weight'] * match['confidence']
                            used_keywords.add(match['keyword'])
                        else:
                            # 충돌 있음 - 가장 적합한 특성에 부여
                            best_trait = self._resolve_keyword_conflict(match['keyword'], conflict_traits, all_matches)
                            if best_trait == trait:
                                trait_scores[trait] += match['weight'] * match['confidence']
                                used_keywords.add(match['keyword'])
        
        # 2단계: 문맥 패턴 보너스 적용 (원본 텍스트 사용)
        context_bonuses = self._detect_context_patterns(original_text, morphemes)
        for trait in trait_scores:
            trait_scores[trait] += context_bonuses[trait]
        
        # 3단계: 점수 정규화 및 최종 조정
        return self._normalize_scores(trait_scores)
    
    def _find_keyword_match(self, keyword, morphemes, morpheme_text, original_text):
        """키워드 매칭 여부와 신뢰도 확인"""
        # 1. 정확한 형태소 매칭 (높은 신뢰도)
        if keyword in morphemes:
            return {'match_type': 'exact', 'confidence': 1.0}
        
        # 2. 부분 매칭 (중간 신뢰도)
        for morpheme in morphemes:
            if len(keyword) >= 2 and len(morpheme) >= 2:
                if keyword in morpheme:
                    confidence = len(keyword) / len(morpheme)
                    if confidence >= 0.5:  # 키워드가 형태소의 50% 이상
                        return {'match_type': 'partial', 'confidence': confidence}
                elif morpheme in keyword:
                    confidence = len(morpheme) / len(keyword)
                    if confidence >= 0.7:  # 형태소가 키워드의 70% 이상
                        return {'match_type': 'contains', 'confidence': confidence}
        
        # 3. 복합어 매칭 (낮은 신뢰도)
        if len(keyword) > 3 and keyword in morpheme_text:
            return {'match_type': 'compound', 'confidence': 0.7}
        
        # 4. 원문 매칭 (가장 낮은 신뢰도)
        if keyword in original_text:
            return {'match_type': 'text', 'confidence': 0.5}
        
        return None
    
    def _check_keyword_conflicts(self, keyword, all_matches, used_keywords):
        """키워드 충돌 확인"""
        if keyword in used_keywords:
            return []
        
        conflict_traits = []
        for trait, matches in all_matches.items():
            for match in matches:
                if match['keyword'] == keyword:
                    conflict_traits.append(trait)
        
        return conflict_traits if len(conflict_traits) > 1 else []
    
    def _resolve_keyword_conflict(self, keyword, conflict_traits, all_matches):
        """키워드 충돌 해결 - 가장 적합한 특성 선택"""
        best_trait = None
        best_score = 0
        
        for trait in conflict_traits:
            for match in all_matches[trait]:
                if match['keyword'] == keyword:
                    # 가중치 * 신뢰도 * 특성 우선순위로 점수 계산
                    trait_priority = {"정직": 1.0, "열정": 0.9, "창의": 0.8, "존중": 0.7}
                    score = match['weight'] * match['confidence'] * trait_priority.get(trait, 0.5)
                    
                    if score > best_score:
                        best_score = score
                        best_trait = trait
        
        return best_trait
    
    def _detect_context_patterns(self, text, morphemes):
        """문맥 패턴 감지 및 가중치 조정"""
        context_bonuses = {"정직": 0, "열정": 0, "창의": 0, "존중": 0}
        text_lower = text.lower()
        
        # 1. 제안/아이디어 표현 패턴 (창의 강화)
        proposal_patterns = [
            "제안해", "제안해볼", "제안하면", "아이디어", "방법을", "방안을", 
            "~하면 어떨까", "~하는 건 어떨까", "~해보면", "~시도해", "생각해낸",
            "떠올린", "구상해", "기획해", "설계해", "틀을", "기존 틀", "완전히 깬", 
            "안 해본", "시도한다는", "혁신적", "참신", "색다른"
        ]
        
        creative_indicators = 0
        for pattern in proposal_patterns:
            if pattern in text_lower:
                creative_indicators += 1
        
        # 추가: 창의적 찬사 패턴
        if any(word in text_lower for word in ["기발", "참신", "독특", "색다른", "혁신적"]):
            creative_indicators += 2
        
        if creative_indicators >= 1:
            context_bonuses["창의"] += 20 * creative_indicators
        
        # 2. 열정적 의지 표현 패턴 (열정 강화)
        enthusiasm_patterns = [
            "열심히", "최선을", "노력", "도전", "의욕", "대단해", "좋아해", "푹 빠져",
            "진짜 좋아", "정말 인상", "~하겠습니다", "~할게요", "~하고 싶어", "열의가",
            "진심으로 생각", "진심으로 임하", "열의", "대단하다", "진심어린",
            "빠져있는", "빠져서", "좋아하는 게", "정말 좋아", "완전 좋아"
        ]
        
        enthusiasm_indicators = 0
        for pattern in enthusiasm_patterns:
            if pattern in text_lower:
                enthusiasm_indicators += 1
        
        # 추가: "진심으로 + 감정/행동" 패턴 (열정 강화)
        if "진심으로" in text_lower and any(word in text_lower for word in ["생각", "느껴", "임하", "대하"]):
            enthusiasm_indicators += 2  # 진심으로 + 행동은 열정 강화
        
        if enthusiasm_indicators >= 1:
            context_bonuses["열정"] += 15 * enthusiasm_indicators
        
        # 3. 배려/소통 표현 패턴 (존중 강화)
        respect_patterns = [
            "함께", "서로", "배려", "이해하", "공감", "소통", "협력", "도움",
            "고객 입장", "~님의", "~분의", "어떻게 생각", "의견"
        ]
        
        respect_indicators = 0
        for pattern in respect_patterns:
            if pattern in text_lower:
                respect_indicators += 1
        
        if respect_indicators >= 2:  # 존중은 2개 이상일 때 보너스
            context_bonuses["존중"] += 10 * respect_indicators
        
        # 4. 솔직함/현실성 표현 패턴 (정직 강화) - 명시적 키워드 제외
        honesty_patterns = [
            "솔직히", "사실", "현실적으로", "실제로", "진짜로", "정말로",
            "인정", "털어놓", "고백", "사과"
        ]
        
        # 단, 창의적 제안과 함께 나오면 창의 우선
        if creative_indicators == 0:  # 창의 표현이 없을 때만 정직 보너스
            honesty_indicators = 0
            for pattern in honesty_patterns:
                if pattern in text_lower:
                    honesty_indicators += 1
            
            if honesty_indicators >= 1:
                context_bonuses["정직"] += 10 * honesty_indicators
        
        return context_bonuses
    
    def _normalize_scores(self, trait_scores):
        """점수 정규화 및 최종 조정"""
        # 점수를 0-100 범위로 정규화
        max_possible_score = sum(self.score_weights.values()) * 3  # 여유있게 설정
        
        normalized_scores = {}
        for trait, score in trait_scores.items():
            normalized_score = min(100, (score / max_possible_score) * 100)
            normalized_scores[trait] = max(0, int(normalized_score))
        
        return normalized_scores
    
    def _detect_explicit_traits(self, text):
        """명시적 인재상 키워드 검출 (정직, 창의, 열정, 존중) - 문맥 고려"""
        # 명시적 인재상 키워드들
        explicit_keywords = {
            "정직": ["정직"],
            "창의": ["창의"],  
            "열정": ["열정"],
            "존중": ["존중"]
        }
        
        # 텍스트를 소문자로 변환하여 매칭
        text_lower = text.lower()
        
        # 창의적 제안 표현이 있는지 먼저 확인
        creative_proposal_patterns = ["제안해", "아이디어", "방법을", "방안을", "하면 어떨까", "해보면"]
        has_creative_proposal = any(pattern in text_lower for pattern in creative_proposal_patterns)
        
        # 각 키워드의 첫 번째 출현 위치 찾기
        keyword_positions = []
        
        for trait, keywords in explicit_keywords.items():
            for keyword in keywords:
                pos = text_lower.find(keyword)
                if pos != -1:  # 키워드가 발견됨
                    # 만약 창의적 제안 표현이 있고 "정직"이 부차적으로 언급된 경우 무시
                    if has_creative_proposal and trait == "정직" and any(pattern in text_lower for pattern in creative_proposal_patterns):
                        # "정직하게" 같은 부사적 표현인지 확인
                        if pos > 0 and text_lower[pos-1:pos+len(keyword)+1].endswith("하게"):
                            continue  # 부사적 용법이면 명시적 키워드로 보지 않음
                    
                    keyword_positions.append((pos, trait, keyword))
        
        # 위치가 발견된 키워드가 있으면 가장 앞에 나오는 것 선택
        if keyword_positions:
            # 위치순으로 정렬하여 첫 번째 것 선택
            keyword_positions.sort(key=lambda x: x[0])
            first_pos, first_trait, first_keyword = keyword_positions[0]
            
            self.logger.info(f"명시적 인재상 키워드 '{first_keyword}' 검출됨 (위치: {first_pos}) -> {first_trait}")
            return first_trait
        
        return None
    
    def _calculate_keyword_score(self, morphemes, trait):
        """개별 특성에 대한 키워드 점수 계산 (하위 호환성 유지)"""
        all_scores = self._calculate_all_keyword_scores(morphemes)
        return all_scores.get(trait, 0)
    
    def _calculate_similarity_score(self, morphemes, trait):
        """Word2Vec를 사용한 개선된 유사도 계산"""
        if not self.word2vec_model:
            return 0.0
        
        # 모든 레벨의 키워드 사용 (핵심 > 복합 > 일반 순으로 가중치)
        all_keywords = []
        keyword_weights = {}
        
        for level, keywords in self.cj_keywords[trait].items():
            weight = self.score_weights[level] / 35  # 정규화
            for keyword in keywords:
                all_keywords.append(keyword)
                keyword_weights[keyword] = weight
        
        max_weighted_similarity = 0.0
        best_match = ""
        
        try:
            for morpheme in morphemes:
                # Gensim Word2Vec 모델의 경우 wv 속성 사용
                if hasattr(self.word2vec_model, 'wv'):
                    model_vocab = self.word2vec_model.wv
                else:
                    model_vocab = self.word2vec_model
                
                if morpheme in model_vocab:
                    for keyword in all_keywords:
                        if keyword in model_vocab:
                            similarity = model_vocab.similarity(morpheme, keyword)
                            # 키워드 중요도에 따른 가중치 적용
                            weighted_similarity = similarity * keyword_weights[keyword]
                            
                            if weighted_similarity > max_weighted_similarity:
                                max_weighted_similarity = weighted_similarity
                                best_match = f"{morpheme}->{keyword}"
            
            # 로깅
            if max_weighted_similarity > 0:
                self.logger.debug(f"최고 유사도 매치: {best_match} ({max_weighted_similarity:.3f})")
            
            return max_weighted_similarity
            
        except Exception as e:
            self.logger.error(f"Word2Vec 유사도 계산 중 오류: {e}")
            return 0.0
    
    def classify(self, text, user_id):
        """메시지 분류 메인 함수 (우선순위: 명시적 인재상 → 키워드 매칭 → 유사도 계산)"""
        # 전처리: 빈 메시지 처리
        if not text or len(text.strip()) < 2:
            return {
                "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                "primary_trait": "무응답",
                "summary": "메시지가 너무 짧거나 비어있음",
                "method": "skip",
                "confidence": 0.0,
                "morphemes": []
            }
        
        # 0단계: 명시적 인재상 키워드 검출 (최우선)
        explicit_trait = self._detect_explicit_traits(text)
        if explicit_trait:
            morphemes = self._analyze_morphemes(text)  # 형태소도 포함
            return {
                "cj_values": {trait: 100 if trait == explicit_trait else 0 for trait in ["정직", "열정", "창의", "존중"]},
                "primary_trait": explicit_trait,
                "summary": f"명시적 인재상 키워드 '{explicit_trait}' 검출",
                "method": "explicit_trait",
                "confidence": 1.0,
                "morphemes": morphemes
            }
        
        # 1단계: 형태소 분석
        morphemes = self._analyze_morphemes(text)
        
        if not morphemes:
            return {
                "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                "primary_trait": "분석실패",
                "summary": "형태소 분석 결과가 없음",
                "method": "analysis_failed",
                "confidence": 0.0,
                "morphemes": []
            }
        
        # 2단계: 스마트 키워드 매칭 시도 (중복 해결)
        keyword_scores = self._calculate_all_keyword_scores(morphemes, text)
        total_keyword_score = sum(keyword_scores.values())
        
        # 키워드 매칭 성공시 즉시 반환
        if total_keyword_score > 0:
            primary_trait = max(keyword_scores, key=keyword_scores.get)
            confidence = keyword_scores[primary_trait] / 100.0
            
            return {
                "cj_values": keyword_scores,
                "primary_trait": primary_trait,
                "summary": f"키워드 매칭으로 '{primary_trait}' 특성 분류됨",
                "method": "keyword_matching",
                "confidence": confidence,
                "morphemes": morphemes
            }
        
        # 3단계: Word2Vec 유사도 평가
        if self.word2vec_model:
            similarity_scores = {}
            for trait in ["정직", "열정", "창의", "존중"]:
                similarity_scores[trait] = self._calculate_similarity_score(morphemes, trait)
            
            max_similarity = max(similarity_scores.values())
            
            # 임계값 검사
            if max_similarity >= self.similarity_threshold:
                # 유사도 점수를 0-100 스케일로 변환
                final_scores = {trait: int(score * 100) for trait, score in similarity_scores.items()}
                best_trait = max(similarity_scores, key=similarity_scores.get)
                
                return {
                    "cj_values": final_scores,
                    "primary_trait": best_trait,
                    "summary": f"Word2Vec 유사도로 '{best_trait}' 특성 분류됨 (유사도: {max_similarity:.3f})",
                    "method": "word2vec_similarity",
                    "confidence": max_similarity,
                    "morphemes": morphemes
                }
        else:
            max_similarity = 0.0
        
        # 4단계: 분류 실패
        return {
            "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
            "primary_trait": "분류실패",
            "summary": f"키워드 매칭 및 유사도 분석 모두 임계값 미달 (최대 유사도: {max_similarity:.3f})",
            "method": "classification_failed",
            "confidence": max_similarity,
            "morphemes": morphemes
        }
    
    def get_user_profile(self, user_id, messages):
        """사용자의 전체 발언을 종합하여 프로필 생성"""
        if not messages:
            return {"error": "분석할 메시지가 없습니다"}
        
        # 각 메시지 분류
        classifications = []
        method_counts = {
            "explicit_trait": 0,
            "keyword_matching": 0, 
            "word2vec_similarity": 0, 
            "classification_failed": 0,
            "skip": 0,
            "analysis_failed": 0
        }
        
        for msg in messages:
            result = self.classify(msg["text"], user_id)
            classifications.append(result)
            method = result.get("method", "unknown")
            if method in method_counts:
                method_counts[method] += 1
        
        # 평균 점수 계산 (분류 성공한 메시지만)
        avg_cj_values = {}
        for value in ["정직", "열정", "창의", "존중"]:
            scores = [c["cj_values"][value] for c in classifications if c["cj_values"][value] > 0]
            avg_cj_values[value] = round(sum(scores) / len(scores)) if scores else 0
        
        # 가장 강한 특성들 찾기
        top_traits = sorted(avg_cj_values.items(), key=lambda x: x[1], reverse=True)[:2]
        
        # 분류 성공률 계산
        total_messages = len(messages)
        successful_classifications = method_counts["explicit_trait"] + method_counts["keyword_matching"] + method_counts["word2vec_similarity"]
        success_rate = successful_classifications / total_messages if total_messages > 0 else 0
        
        return {
            "user_id": user_id,
            "message_count": total_messages,
            "avg_cj_values": avg_cj_values,
            "top_traits": [trait for trait, _ in top_traits],
            "classification_methods": method_counts,
            "success_rate": round(success_rate * 100, 1),
            "overall_summary": f"{user_id}님의 주요 특성: {', '.join([trait for trait, _ in top_traits[:2]])}"
        }