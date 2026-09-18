"""Financial education classifier retaining the existing HTTP field contract."""
import json
import os
from app.financial_profile import AXES, PROFILE_VERSION, FACILITATOR_PROMPT, RUBRIC_PROMPT, normalized_scores, unassessed_result

class MessageClassifierGPT:
    def __init__(self, client=None):
        self.client = client
        self.model = os.getenv('CLASSIFIER_MODEL', 'gpt-4o-mini')
        if self.client is None and os.getenv('OPENAI_API_KEY'):
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=20.0, max_retries=0)

    def _create_system_prompt(self):
        return FACILITATOR_PROMPT + '\n발언 분류 기준 초안:\n' + RUBRIC_PROMPT + '''
각 축은 0~100 정수다. 실제 발언에 드러난 근거만 평가하고 단어 등장만으로 높은 점수를 주지 마세요.
없는 근거는 0으로 표시합니다. 다음 JSON 형식으로만 응답하세요:
{"cj_values":{"금융이해":0,"위험인식":0,"계획성":0,"실천의지":0},"summary":"발언 근거를 설명"}'''

    def classify(self, text, user_id, context=None):
        if not text or not text.strip():
            return {'cj_values':dict.fromkeys(AXES,0),'primary_trait':'무응답','summary':'평가할 발언이 없습니다.','evaluation_status':'no_participation','profile_version':PROFILE_VERSION}
        if self.client is None:
            return unassessed_result()
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0.2, max_tokens=400,
                response_format={'type':'json_object'},
                messages=[{'role':'system','content':self._create_system_prompt()},
                          {'role':'user','content':json.dumps({'user_id':user_id,'text':text,'context':context or {}},ensure_ascii=False)}])
            raw = json.loads(response.choices[0].message.content)
            if not isinstance(raw.get('cj_values'),dict) or not all(axis in raw['cj_values'] for axis in AXES):
                return unassessed_result()
            scores = normalized_scores(raw['cj_values'])
            primary = max(scores,key=scores.get) if max(scores.values()) else '해당없음'
            return {'cj_values':scores,'primary_trait':primary,'summary':str(raw.get('summary','')),
                    'evaluation_status':'assessed','profile_version':PROFILE_VERSION}
        except Exception as error:
            body = getattr(error, 'body', {}) or {}
            detail = body.get('error', body) if isinstance(body, dict) else {}
            code = detail.get('code') if isinstance(detail, dict) else None
            reason = ('insufficient_quota' if code in ('insufficient_quota', 'credit_balance_exhausted')
                      else 'rate_limit_exceeded' if code == 'rate_limit_exceeded'
                      else 'provider_unavailable')
            return {**unassessed_result(), 'provider_error': reason}

    def get_user_profile(self, user_id, messages):
        results = [self.classify(m.get('text','') if isinstance(m,dict) else str(m),user_id) for m in messages]
        assessed = [r for r in results if r.get('evaluation_status')=='assessed']
        values = {axis:round(sum(r['cj_values'][axis] for r in assessed)/len(assessed)) if assessed else 0 for axis in AXES}
        return {'user_id':user_id,'message_count':len(messages),'avg_cj_values':values,
                'top_traits':[axis for axis in sorted(AXES,key=lambda a:values[a],reverse=True) if values[axis]>0][:2],
                'overall_summary': '평가된 발언만 집계했습니다.' if assessed else 'AI 평가 결과가 없습니다.'}
