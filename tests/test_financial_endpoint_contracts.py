"""Exercise all production parsers/response models while replacing only the model boundary."""
import os
os.environ['OPENAI_API_KEY']='test-not-a-real-key'
import json,unittest
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
import app.main as service
from app.financial_profile import AXES

class Completion:
 def __init__(self,value):self.value=value
 def create(self,**kwargs):
  text=self.value if isinstance(self.value,str) else json.dumps(self.value,ensure_ascii=False)
  return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
class AsyncCompletion(Completion):
 async def create(self,**kwargs):return super().create(**kwargs)
def sdk(value,asynchronous=False):
 return SimpleNamespace(chat=SimpleNamespace(completions=(AsyncCompletion if asynchronous else Completion)(value)))

class EndpointContractTests(unittest.TestCase):
 def setUp(self):self.client=TestClient(service.app)
 def test_action_and_reason_objects_preserve_feedback_text_contract(self):
  value={'overall_score':60,'cj_trait_scores':dict.fromkeys(AXES,60),'participation_summary':'계획을 설명함','strengths':[],
         'improvements':[{'improvement':'기간을 30개월로 조정해 보세요.','reason':'월 10만 원씩 300만 원을 모으려면 30개월이 필요합니다.'}],
         'personalized_feedback':'금액과 기간을 함께 비교해 보세요.','top_messages':[]}
  with patch.object(service.discussion_evaluator,'client',sdk(value)):
   r=self.client.post('/evaluate',json={'user_id':'test','user_messages':[{'text':'300만 원을 모으려고 합니다.'}]})
  self.assertEqual(r.status_code,200,r.text)
  self.assertEqual(r.json()['improvements'],['기간을 30개월로 조정해 보세요. 월 10만 원씩 300만 원을 모으려면 30개월이 필요합니다.'])
 def test_health_declares_configuration_separately_from_connectivity(self):
  body=self.client.get('/health').json()
  self.assertEqual(body['status'],'ok')
  self.assertEqual(body['provider_connectivity'],'not_checked_by_health')
 def test_personal_evaluation_response(self):
  value={'overall_score':60,'cj_trait_scores':dict.fromkeys(AXES,60),'participation_summary':'계획을 설명함','strengths':['금액과 기간 명시'],'improvements':[],'personalized_feedback':'실행 기준을 정해보세요.','top_messages':['월 10만원을 모으겠습니다.']}
  with patch.object(service.discussion_evaluator,'client',sdk(value)):
   r=self.client.post('/evaluate',json={'user_id':'test','user_messages':[{'text':'월 10만원을 모으겠습니다.'}]})
  self.assertEqual(r.status_code,200,r.text)
  self.assertEqual(r.json()['overall_score'],60)
  self.assertEqual(r.json()['evaluation_method'],'GPT 기반 개인 맞춤 평가')
 def test_overall_response_counts_real_participants(self):
  with patch.object(service.discussion_evaluator,'client',sdk({'discussion_summary':'두 참여자가 저축 목표를 설명했습니다.'})):
   r=self.client.post('/discussion-overall',json={'all_user_messages':[{'nickname':'one','text':'목표1'},{'nickname':'two','text':'목표2'}]})
  self.assertEqual(r.status_code,200,r.text)
  self.assertEqual(r.json()['total_participants'],2)
 def test_user_summary_resolves_only_target_speakers_message_ids(self):
  value={'user_id':'test','topics':[{'topic':'목표','relevance_score':.8,'related_message_ids':[1,2],'summary':'목표를 설명함'}]}
  with patch.object(service.discussion_summarizer,'async_client',sdk(value,True)),patch.object(service.discussion_summarizer,'_cache',{}):
   r=self.client.post('/user-summary',json={'user_id':'test','discussion_topics':[{'name':'목표'}],'chat_history':[{'nickname':'test','text':'내 목표'},{'nickname':'other','text':'다른 사람 목표'}]})
  self.assertEqual(r.status_code,200,r.text)
  self.assertEqual(r.json()['topics'][0]['related_statements'],['내 목표'])
 def test_encouragement_response(self):
  with patch.object(service.participant_monitor,'client',sdk('test님, 편하게 의견을 나눠주세요.')):
   r=self.client.post('/encouragement',json={'nickname':'test','chat_history':[]})
  self.assertEqual(r.status_code,200,r.text)
  self.assertEqual(r.json()['nickname'],'test')
  self.assertTrue(r.json()['message'])
 def test_profile_uses_assessed_scores(self):
  with patch.object(service.message_classifier,'client',sdk({'cj_values':dict.fromkeys(AXES,60),'summary':'발언 근거'})):
   r=self.client.post('/profile',json={'user_id':'test','messages':[{'text':'금액을 정합니다'}]})
  self.assertEqual(r.status_code,200,r.text)
  self.assertEqual(r.json()['avg_cj_values'],dict.fromkeys(AXES,60))
