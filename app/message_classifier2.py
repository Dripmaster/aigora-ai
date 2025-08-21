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
    
    def __init__(self, similarity_threshold=0.4):
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
        
        # CJ 인재상 형태소 기반 키워드 정의 (대폭 확장된 키워드 세트)
        self.cj_keywords = {
            "정직": {
                "핵심": ["솔직", "사실", "진실", "정직", "투명", "정확", "현실", "실제", "실상", "진심", 
                        "솔직하다", "사실이다", "진실하다", "정직하다", "투명하다", "성실하다", "진정하다",
                        "올바르다", "바르다", "옳다", "맞다", "정확하다", "확실하다"],
                "복합": ["솔직히", "사실대로", "정직하게", "투명하게", "거짓없이", "진실하게", "정말로", "진짜로",
                        "솔직히말하면", "사실을말하면", "진실을말하면", "있는그대로", "거짓말하지", "속이지않고",
                        "정직하게말씀드리면", "사실대로말씀드리면", "진심으로말씀드리면", "털어놓고말하면",
                        "솔직히말씀드리면", "진심어린", "정말진심으로", "거짓없이말하면", "있는그대로말하면",
                        "사실을인정하면", "현실적으로말하면", "실제상황은", "정확한사실은", "정말로그렇다면"],
                "일반": ["확실", "분명", "성실", "신뢰", "믿음", "인정", "명확", "객관", "직접", "명료",
                        "신뢰하다", "믿다", "확신하다", "명확하다", "분명하다", "확실하다", "객관적이다",
                        "인정하다", "고백하다", "털어놓다", "시인하다", "받아들이다", "인간적으로",
                        "정확히", "확인하다", "증명하다", "입증하다", "증언하다", "보장하다", "약속하다",
                        "신용", "신의", "믿을만하다", "믿음직하다", "신뢰성", "신뢰감", "안심", "확신"],
                "보조": ["참", "진정", "명백", "털어놓", "고백", "사과", "양심", "떳떳", "정말", "진짜",
                        "진심", "양심적", "도덕적", "윤리적", "깨끗하게", "떳떳하게", "당당하게", "부끄럽지않게",
                        "정의", "올바름", "바름", "착하다", "선하다", "고결하다", "깨끗하다", "순수하다"]
            },
            "열정": {
                "핵심": ["열정", "적극", "최선", "열심", "도전", "전력", "의욕", "에너지", "헌신", "활기",
                        "열정적이다", "적극적이다", "의욕적이다", "활발하다", "역동적이다", "도전적이다",
                        "불타오르다", "뜨겁다", "강렬하다", "치열하다", "집중하다", "몰두하다", "투지"],
                "복합": ["열정적으로", "적극적으로", "최선을다해", "열심히", "전력을다해", "헌신적으로", "의욕적으로",
                        "최선을다하겠습니다", "열심히하겠습니다", "적극적으로참여", "도전해보겠습니다", 
                        "전력으로임하겠습니다", "최대한노력하겠습니다", "혼신의힘을다해", "열의를가지고",
                        "죽어라고", "죽도록", "최대한", "한계까지", "끝까지", "포기하지않고", "열심히해보겠습니다",
                        "최선을다해서", "전력투구하겠습니다", "온힘을다해", "혼신을다해", "전심전력으로"],
                "일반": ["노력", "성장", "발전", "활력", "동기", "추진", "의지", "끈기", "집중", "몰입", "행동",
                        "노력하다", "성장하다", "발전하다", "향상하다", "개선하다", "달성하다", "실현하다",
                        "진취적", "능동적", "자발적", "주도적", "실행력", "추진력", "행동력", "실천력",
                        "격려", "독려", "박차", "가속", "촉진", "강화", "증진", "도약", "비약", "약진",
                        "발전시키다", "향상시키다", "완성하다", "이루다", "성취하다", "쟁취하다", "획득하다"],
                "보조": ["힘내", "파이팅", "화이팅", "각오", "시도", "실행", "좋", "훌륭", "대단", "멋있",
                        "열심", "부지런", "성실", "근면", "꾸준", "지속", "계속", "더욱", "한층더", "더더욱",
                        "투혼", "투지", "기개", "기백", "용기", "담력", "대담", "과감", "용감", "씩씩",
                        "활발", "활동적", "생동감", "활력", "기운", "정신", "의욕", "기력", "원기"]
            },
            "창의": {
                "핵심": ["혁신", "아이디어", "독창", "창의", "창조", "기발", "참신", "독특", "상상", "발명",
                        "혁신적이다", "창의적이다", "독창적이다", "창조적이다", "기발하다", "참신하다",
                        "영감", "직감", "통찰", "발상", "착상", "구상", "기획", "설계", "디자인"],
                "복합": ["혁신적으로", "창의적으로", "독창적으로", "새로운방법", "아이디어제안", "참신하게",
                        "새로운아이디어로", "혁신적인방법으로", "창의적인접근으로", "독창적인해결책으로",
                        "아이디어를제안해", "새로운관점에서", "다른시각으로", "색다른방법으로",
                        "창의적발상으로", "혁신적사고로", "독창적아이디어로", "새로운시도로", "참신한발상으로",
                        "기발한생각으로", "새로운접근방식으로", "다양한방법으로", "여러가지시도로"],
                "일반": ["새로운", "개선", "변화", "개혁", "차별", "개발", "창작", "발견", "개척", "실험",
                        "새롭다", "개선하다", "변화하다", "개혁하다", "차별화하다", "개발하다", "발견하다",
                        "혁명적", "획기적", "변화무쌍", "다양하다", "여러가지", "다채롭다", "풍부하다",
                        "신기하다", "놀랍다", "흥미롭다", "재미있다", "독특하다", "특별하다", "색다르다",
                        "변신", "변모", "전환", "변혁", "진화", "발전", "도약", "진보", "개선점", "향상안"],
                "보조": ["바꿔", "새로", "신선", "특별", "색다른", "기발", "다른", "특이", "신", "톡톡",
                        "바뀐", "다르게", "특별하게", "독특하게", "재미있게", "흥미롭게", "신기하게", "놀랍게",
                        "새", "신규", "신개념", "신선함", "참신함", "톡톡튀는", "유니크", "오리지널", "첫"]
            },
            "존중": {
                "핵심": ["배려", "존중", "경청", "공감", "이해", "사려", "친절", "따뜻", "포용", "겸손",
                        "배려하다", "존중하다", "경청하다", "공감하다", "이해하다", "친절하다", "겸손하다",
                        "관심", "관용", "포용", "수용", "받아들이다", "아껴주다", "소중히여기다"],
                "복합": ["함께협력", "서로도움", "배려하는", "고객입장", "존중하며", "이해하려", "공감하는",
                        "고객입장에서생각", "서로존중하며", "배려하는마음으로", "이해하려는노력으로",
                        "공감하는마음으로", "함께소통하며", "협력적인자세로", "상호이해를통해",
                        "고객만족을위해", "고객을위한", "고객중심으로", "고객의마음으로", "고객관점에서",
                        "서로를위해", "모두를위해", "함께노력해서", "협력하여", "도움을주며", "힘을합쳐"],
                "일반": ["소통", "협력", "팀워크", "도움", "지원", "협조", "화합", "조화", "매너", "예의",
                        "소통하다", "협력하다", "도움주다", "지원하다", "협조하다", "화합하다", "조화롭다",
                        "예의바르다", "정중하다", "친근하다", "상냥하다", "다정하다", "온화하다", "부드럽다",
                        "친화력", "사교성", "인화", "화목", "단합", "결속", "유대", "신뢰관계", "인간관계",
                        "예절", "에티켓", "예의범절", "공손", "겸양", "겸허", "사양", "양보", "배려심"],
                "보조": ["같이", "함께", "서로", "도와", "마음", "감정", "입장", "고객", "대화", "이야기",
                        "서로서로", "모두함께", "다같이", "상대방", "상호간", "양쪽모두", "서로의", "상대의",
                        "손님", "방문객", "이용객", "고객님", "회원", "사용자", "소비자", "구매자", "관계자",
                        "동료", "팀원", "구성원", "멤버", "파트너", "협력자", "동반자", "상대", "타인"]
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
    
    def _calculate_all_keyword_scores(self, morphemes):
        """모든 특성에 대한 키워드 점수를 동시에 계산하여 중복 해결"""
        # 전체 키워드 매칭 결과 저장
        all_matches = {}
        trait_scores = {"정직": 0, "열정": 0, "창의": 0, "존중": 0}
        
        # 형태소 텍스트 생성
        morpheme_text = "".join(morphemes)
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
        
        # 2단계: 점수 정규화 및 최종 조정
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
        """명시적 인재상 키워드 검출 (정직, 창의, 열정, 존중) - 텍스트 내 첫 번째 출현 순서 우선"""
        # 명시적 인재상 키워드들
        explicit_keywords = {
            "정직": ["정직"],
            "창의": ["창의"],  
            "열정": ["열정"],
            "존중": ["존중"]
        }
        
        # 텍스트를 소문자로 변환하여 매칭
        text_lower = text.lower()
        
        # 각 키워드의 첫 번째 출현 위치 찾기
        keyword_positions = []
        
        for trait, keywords in explicit_keywords.items():
            for keyword in keywords:
                pos = text_lower.find(keyword)
                if pos != -1:  # 키워드가 발견됨
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
        keyword_scores = self._calculate_all_keyword_scores(morphemes)
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