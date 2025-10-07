import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

class MessageClassifier:
    """
    CJ 식음 서비스 매니저 교육용 메시지 분류기
    토론 참여자의 발언을 CJ 인재상과 성격 특성별로 분류하여 대시보드 분석용 데이터 제공
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.gpt_enabled = True
            self.model = "gpt-4o-mini"
            print(f"MessageClassifier: OpenAI API 키 설정 완료")
        else:
            self.gpt_enabled = False
            print("MessageClassifier: OpenAI API 키 설정 실패 - 룰 기반 분류 사용")

        # CJ 인재상 키워드 정의 (가중치별) - 최적화된 키워드 세트
        self.cj_keywords = {
            "정직": {
                "핵심": ["솔직히", "사실대로", "진실을", "정직하게", "솔직하게", "거짓없이", "있는 그대로", "진솔하게", "투명하게", "정직히", "솔직한", "떳떳하게", "양심적으로", "사실", "실제로", "진짜로", "정말로", "현실적으로"],
                "복합": ["솔직히 말하면", "사실대로 말하면", "진실을 말하면", "정직하게 고백하면", "솔직하게 얘기하자면", "거짓 없이 말하면", "있는 그대로 말하면", "투명한 소통", "정확한 정보", "명백한 사실", "아니라고 봐요", "아니라고 생각", "속이는 건 아니", "허위광고", "거짓말", "속임수", "정직하게 말씀드리면", "사실을 말하면", "실상을 보면", "현실적으로 말하면"],
                "일반": ["투명한", "정확한", "확실한", "분명한", "진실한", "진실", "성실한", "진정한", "직접적인", "명료한", "객관적인", "신뢰할 수 있는", "믿을 만한", "회사 잘못 인정", "사과", "진심", "봐요", "생각해요", "생각합니다", "실상", "현실", "사실상", "실제", "정말", "진짜", "참", "인정", "털어놓", "고백"],
                "보조": ["참으로", "진정", "명백히", "실은", "실상은", "확실하게", "분명하게", "투명하게", "솔직", "정직", "진실", "사실", "실제", "현실", "정말", "진짜"]
            },
            "열정": {
                "핵심": ["열정적으로", "적극적으로", "최선을 다해", "열심히", "도전하겠", "전력을 다해", "의욕적으로", "에너지 넘치게", "불타는 마음으로", "뜨겁게", "열렬히", "활기차게", "헌신적으로", "열의를 다해", "혼신의 힘으로", ],
                "복합": ["최선을 다하겠습니다", "적극적으로 참여", "열심히 노력", "도전해보겠", "전력으로 임하겠", "최선을 다해 노력", "열정을 다해", "의욕을 가지고", "활력있게 임하겠", "에너지를 쏟아", "최선을 다하는 자세로", "헌신적인 자세로", "열정적인 마음으로", "적극적인 자세로"],
                "일반": ["헌신적인", "노력하는", "도전", "성장", "발전", "노력", "의욕", "활력", "동기", "추진력", "의지", "끈기", "집중", "몰입", "행동력", "실천력", "진취적", "의욕적", "능동적", "자발적", "주도적", "실행력", "추진", "실천", "노력", "성장", "발전", "개선", "향상", "달성"],
                "보조": ["열심", "힘내", "파이팅", "화이팅", "각오", "의욕", "활기", "에너지", "추진", "실행", "도전", "시도", "노력", "힘", "기운", "정신", "마음", "의지", "의욕", "열의", "대단", "멋있", "좋", "잘", "훌륭", "완전", "최고"]
            },
            "창의": {
                "핵심": ["혁신적인", "새로운 아이디어", "독창적인", "창의적인", "혁신", "기발한", "참신한", "독특한", "창조적인", "발명적인", "예술적인", "상상력", "혁신적", "창의적", "독창적", "창조적", "혁명적", "제안"],
                "복합": ["새로운 방법으로", "아이디어를 제안", "혁신적으로 접근", "창의적으로 생각", "새로운 관점에서", "독창적으로 접근", "창조적으로 해결", "기발한 아이디어로", "참신하게 접근", "새로운 아이디어를 제안해드리겠습니다", "혁신적인 방법", "창의적인 접근", "독창적인 해결책", "새로운 시각", "차별화된 방식"],
                "일반": ["새로운", "아이디어", "개선", "변화", "개혁", "차별화", "개발", "창작", "발견", "발명", "개척", "시도", "실험", "기획", "설계", "구상", "혁신", "창의", "독창", "창조", "발명", "개발", "변화", "개선", "개혁", "혁명", "진화", "발전", "신기술", "신방법", "신개념"],
                "보조": ["바꿔", "바뀐", "새로", "신선한", "독특", "특별한", "색다른", "참신", "기발", "톡톡한", "다른", "다르게", "색다른", "특이한", "신선", "새", "신", "참신", "기발", "독특", "특별", "차별", "차이", "변화", "개선"]
            },
            "존중": {
                "핵심": ["배려하는", "존중하는", "경청하는", "공감하는", "이해하려고", "사려깊은", "친절한", "따뜻한", "포용하는", "관용적인", "겸손한", "예의바른", "배려심", "존중심", "공감능력", "이해심", "입장에서", "마음을", "고객 마음", "소통을 통해", "상대방 마음", "고객의 입장"],
                "복합": ["함께 협력", "서로 도움", "의견을 듣고", "고객 입장에서 생각", "배려하는 마음", "서로 존중하며", "배려심을 갖고", "이해하려는 노력", "공감하는 마음으로", "함께 소통하며", "협력적인 자세", "배려하는 자세", "존중하는 마음", "이해하는 마음", "소통하는 자세", "고객을 위해", "고객 감정을", "불편함을", "사려깊게 고객의", "함께 노력해서", "고객과 회사 사이의", "오해를 풀어가는"],
                "일반": ["소통", "협력", "팀워크", "공감", "경청", "도움", "배려", "이해", "지원", "협조", "화합", "조화", "친화", "매너", "예의", "친근", "협업", "상호작용", "상호이해", "상호존중", "상호배려", "커뮤니케이션", "대화", "토론", "논의", "상담", "조언", "관용적", "친절하게", "예의바르게", "겸손하게", "사려깊게", "오해", "문제를 해결"],
                "보조": ["같이", "함께", "서로", "도와", "배려", "친절", "예의", "매너", "협력", "지원", "도움", "공감", "이해", "소통", "대화", "이야기", "얘기", "말씀", "의견", "생각", "어떻게", "어떨까", "어떤", "무엇", "왜", "어디서", "언제", "누구", "불편", "감정", "분노", "화난", "진정", "달래", "자세"]
            }
        }
        
        # 부정어 패턴
        self.negation_patterns = [
            "않", "없", "못", "안", "아니", "말고", "빼고"
        ]
        
        # 점수 가중치 (핵심 키워드 중시)
        self.score_weights = {
            "핵심": 40,
            "복합": 32, 
            "일반": 20,
            "보조": 10
        }


        # GPT 시스템 프롬프트
        self.system_prompt = """당신은 CJ 식음 서비스 매니저 교육 프로그램의 메시지 분류 전문가입니다.

참여자의 토론 발언을 다음 기준으로 분류해주세요:

**CJ 인재상 (각각 0-100 점수):**
- 정직: 투명하고 진실된 소통을 보여주는 정도
- 열정: 적극적이고 헌신적인 자세를 보여주는 정도  
- 창의: 혁신적이고 새로운 접근을 보여주는 정도
- 존중: 고객과 동료를 배려하는 마음을 보여주는 정도


응답 형식은 반드시 다음 JSON 형태로만 제공하세요:
{
  "cj_values": {
    "정직": 점수,
    "열정": 점수,
    "창의": 점수,
    "존중": 점수
  },
  "primary_trait": "가장_두드러진_CJ_인재상",
  "summary": "한글로_간단한_분류_요약"
}"""

    def classify_with_gpt(self, text: str, user_id: str, context: Optional[Dict] = None) -> Optional[Dict]:
        """GPT를 사용한 메시지 분류"""
        if not self.gpt_enabled:
            return None

        try:
            # 컨텍스트 정보 구성
            context_info = ""
            if context:
                if context.get("previous_messages"):
                    context_info += f"이전 발언들: {context['previous_messages'][-3:]}\n"
                if context.get("discussion_topic"):
                    context_info += f"토론 주제: {context['discussion_topic']}\n"

            user_prompt = f"""{context_info}

분류할 발언: "{text}"
발언자 ID: {user_id}

위 발언을 CJ 인재상 기준으로 분석하여 점수를 매겨주세요."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            print(f"[GPT 분류] {user_id}: {result.get('primary_trait', '분류완료')}")
            return result

        except Exception as e:
            print(f"GPT 분류 오류: {e}")
            return None

    def _has_negation(self, text: str, keyword: str) -> bool:
        """키워드 주변에 부정어가 있는지 확인"""
        keyword_pos = text.find(keyword)
        if keyword_pos == -1:
            return False
        
        # 키워드 앞뒤 8글자 범위에서 부정어 검사 (범위 확대)
        start = max(0, keyword_pos - 8)
        end = min(len(text), keyword_pos + len(keyword) + 8)
        surrounding_text = text[start:end]
        
        # 더 정확하고 확장된 부정어 패턴
        precise_negations = ["안 ", "않", "없", "못", "아니", "말고", "아니라고", "아닌", "아니야", "아니에요", "아닙니다", "안된", "안되", "못해"]
        return any(neg in surrounding_text for neg in precise_negations)
    
    def _calculate_keyword_score(self, text: str, trait: str) -> int:
        """특정 인재상에 대한 키워드 점수 계산"""
        text_lower = text.lower()
        total_score = 0
        matched_keywords = set()  # 중복 방지
        
        keywords_dict = self.cj_keywords[trait]
        
        # 각 가중치 레벨별로 점수 계산
        for level, keywords in keywords_dict.items():
            weight = self.score_weights[level]
            
            for keyword in keywords:
                if keyword in text_lower and keyword not in matched_keywords:
                    # 부정어 체크
                    if self._has_negation(text_lower, keyword):
                        continue  # 부정문이면 점수 부여하지 않음
                    
                    matched_keywords.add(keyword)
                    total_score += weight
        
        # 텍스트 길이에 따른 정규화 (계수 완화)
        text_length = len(text.strip())
        if text_length < 15:
            total_score = int(total_score * 1.1)  # 매우 짧은 텍스트만 가중치 증가
        elif text_length > 100:
            total_score = int(total_score * 0.95)  # 매우 긴 텍스트만 가중치 감소
        
        return min(total_score, 100)  # 최대 100점 제한
    
    def _apply_context_rules(self, text: str, scores: Dict[str, int]) -> Dict[str, int]:
        """강화된 문맥 기반 점수 조정"""
        text_lower = text.lower()
        
        # 1. 강력한 패턴 매칭 (테스트 결과 기반으로 패턴 강화)
        enhanced_patterns = {
            "정직": {
                "strong": ["사실대로", "솔직히", "현실적으로", "정확히", "실상", "실제", "허위광고", "속이는", "거짓말"],
                "medium": ["인정", "사과", "털어놓", "고백", "정말", "진짜", "진심", "분명히"]
            },
            "열정": {
                "strong": ["열정적으로", "전력을 다해", "열심히", "대단해", "최선을 다해", "용기", "도전"],
                "medium": ["노력", "헌신", "활기", "에너지", "의욕", "열의", "멋있", "훌륭", "좋아"]
            },
            "창의": {
                "strong": ["기발한", "참신한", "혁신적", "새로운 아이디어", "다른 회사들은 안 해본", "트렌드 만들 수 있"],
                "medium": ["다른", "색다른", "독특한", "특별한", "창의적", "새로운", "혁신"]
            },
            "존중": {
                "strong": ["소통을 통해서", "고객 입장에서", "사려깊게", "함께 노력해서", "고객과 회사 사이의"],
                "medium": ["배려", "공감", "이해", "소통", "협력", "마음", "입장", "함께", "서로"]
            }
        }
        
        # 강화된 패턴 매칭 및 보너스 적용
        for trait, pattern_groups in enhanced_patterns.items():
            strong_bonus = 0
            medium_bonus = 0
            
            # 강한 패턴 검사
            for pattern in pattern_groups["strong"]:
                if pattern in text_lower:
                    strong_bonus = 15
                    break
            
            # 중간 패턴 검사
            if strong_bonus == 0:  # 강한 패턴이 없을 때만
                for pattern in pattern_groups["medium"]:
                    if pattern in text_lower:
                        medium_bonus = 8
                        break
            
            scores[trait] += strong_bonus + medium_bonus
        
        # 2. 특수 패턴 보너스
        # 질문형 표현 보너스
        question_patterns = ["어떻게 생각", "어떤 의견", "어떨까요", "?", "어떨까", "어떻게"]
        if any(q in text_lower for q in question_patterns):
            scores["존중"] += 8
        
        # 의지 표현 보너스
        intention_words = ["하겠습니다", "하겠어요", "할게요", "하고 싶어요", "만들어보고 싶어요"]
        if any(w in text_lower for w in intention_words):
            scores["열정"] += 8
        
        # 3. 우선순위 규칙 조정 (차이 5점 이하일 때 적용)
        priority = ["정직", "열정", "존중", "창의"]
        max_score = max(scores.values())
        
        if max_score > 0:
            close_traits = [trait for trait, score in scores.items() if max_score - score <= 5]
            if len(close_traits) > 1:
                for trait in priority:
                    if trait in close_traits:
                        scores[trait] += 3
                        break
        
        return {k: min(v, 100) for k, v in scores.items()}
    
    def classify_with_rules(self, text: str, user_id: str) -> Dict:
        """개선된 룰 기반 메시지 분류"""
        if not text or len(text.strip()) < 2:
            return {
                "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                "primary_trait": "무응답",
                "summary": "메시지가 너무 짧거나 비어있음"
            }
        
        # 각 인재상별 점수 계산
        cj_scores = {}
        for trait in ["정직", "열정", "창의", "존중"]:
            cj_scores[trait] = self._calculate_keyword_score(text, trait)
        
        # 문맥 규칙 적용
        cj_scores = self._apply_context_rules(text, cj_scores)
        
        # 모든 점수가 0인 경우 강화된 기본값 부여
        total_score = sum(cj_scores.values())
        if total_score == 0:
            # 강화된 기본 패턴 매칭
            enhanced_basic_patterns = {
                "정직": ["사실", "실제", "정말", "진짜", "사실상", "실상", "현실", "실은", "진심", "인정", "털어놓", "고백"],
                "열정": ["좋", "잘", "훌륭", "대단", "멋있", "최고", "완전", "노력", "도전", "열심", "전력", "의욕", "감동", "전염"],
                "존중": ["생각", "의견", "말씀", "어떻게", "어떨까", "함께", "서로", "고객", "배려", "이해", "공감", "소통", "협력", "대화", "이야기"],
                "창의": ["새로", "다른", "방법", "아이디어", "색다른", "혁신", "참신", "기발", "독특", "특별", "바꿔", "다르게"]
            }
            
            assigned = False
            # 우선순위에 따라 기본값 할당
            for trait in ["정직", "열정", "존중", "창의"]:
                patterns = enhanced_basic_patterns[trait]
                if any(word in text.lower() for word in patterns):
                    cj_scores[trait] = 15  # 기본값 상향 (12 → 15)
                    assigned = True
                    break
            
            if not assigned:
                # 어떤 패턴도 매칭되지 않으면 존중에 기본값
                cj_scores["존중"] = 10
        
        # 주요 특성 찾기
        primary_trait = max(cj_scores, key=cj_scores.get)
        
        return {
            "cj_values": cj_scores,
            "primary_trait": primary_trait,
            "summary": f"'{primary_trait}' 특성이 두드러진 발언으로 분류됨 (성능 향상 룰 기반)"
        }

    def classify(self, text: str, user_id: str, context: Optional[Dict] = None) -> Dict:
        """
        메시지 분류 메인 함수
        
        Args:
            text: 분류할 메시지 텍스트
            user_id: 발언자 ID
            context: 추가 컨텍스트 정보 (이전 메시지, 토론 주제 등)
            
        Returns:
            분류 결과 딕셔너리
        """
        # 빈 메시지 처리
        if not text or len(text.strip()) < 2:
            return {
                "cj_values": {"정직": 0, "열정": 0, "창의": 0, "존중": 0},
                "primary_trait": "무응답",
                "summary": "메시지가 너무 짧거나 비어있음"
            }

        # GPT 분류 시도
        if self.gpt_enabled and context and context.get("use_gpt", True):
            gpt_result = self.classify_with_gpt(text, user_id, context)
            if gpt_result:
                return gpt_result

        # 룰 기반 분류로 백업
        return self.classify_with_rules(text, user_id)

    def get_user_profile(self, user_id: str, messages: List[Dict]) -> Dict:
        """
        사용자의 전체 발언을 종합하여 종합 프로필 생성
        
        Args:
            user_id: 사용자 ID
            messages: 사용자의 모든 메시지 리스트 [{"text": "...", "timestamp": "..."}, ...]
            
        Returns:
            종합 프로필 딕셔너리
        """
        if not messages:
            return {"error": "분석할 메시지가 없습니다"}

        # 각 메시지 분류
        classifications = []
        for msg in messages:
            result = self.classify(msg["text"], user_id)
            classifications.append(result)

        # 평균 점수 계산
        avg_cj_values = {}
        
        for value in ["정직", "열정", "창의", "존중"]:
            scores = [c["cj_values"][value] for c in classifications if c["cj_values"][value] > 0]
            avg_cj_values[value] = round(sum(scores) / len(scores)) if scores else 0

        # 가장 강한 특성들 찾기
        top_traits = sorted(avg_cj_values.items(), key=lambda x: x[1], reverse=True)[:2]

        return {
            "user_id": user_id,
            "message_count": len(messages),
            "avg_cj_values": avg_cj_values,
            "top_traits": [trait for trait, score in top_traits],
            "overall_summary": f"{user_id}님의 주요 특성: {', '.join([trait for trait, score in top_traits])}"
        }