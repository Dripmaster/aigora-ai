import unittest,json
from types import SimpleNamespace
from app.message_classifier_gpt import MessageClassifierGPT
class ClassifierTests(unittest.TestCase):
 def test_provider_failure_cannot_resurrect_cj_keyword_scores(self):
  class Provider:
   def create(self,**kwargs):raise RuntimeError('offline')
  c=MessageClassifierGPT(SimpleNamespace(chat=SimpleNamespace(completions=Provider())))
  result=c.classify('솔직히 정말 좋은 아이디어입니다','student')
  self.assertEqual(sum(result['cj_values'].values()),0)
  self.assertEqual(result['evaluation_status'],'unavailable')
 def test_real_output_adapter_drops_unknown_keys_and_preserves_one_point(self):
  class Provider:
   def create(self,**kwargs):
    payload=json.loads(kwargs['messages'][1]['content'])
    if payload['context']!={'lesson_id':2}:raise AssertionError('lost lesson context')
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'cj_values':{'금융이해':1,'위험인식':60,'계획성':90,'실천의지':0,'정직':100},'summary':'월 예산 근거'})))])
  c=MessageClassifierGPT(SimpleNamespace(chat=SimpleNamespace(completions=Provider())))
  result=c.classify('매달 10만원씩 12개월 모으겠습니다','student',{'lesson_id':2})
  self.assertEqual(result['primary_trait'],'계획성');self.assertEqual(result['cj_values']['금융이해'],1)
  self.assertNotIn('정직',result['cj_values'])
 def test_exhausted_credit_is_distinguished_from_transient_rate_limit(self):
  class QuotaError(Exception):
   body={'code':'credit_balance_exhausted','type':'insufficient_quota'}
  class Provider:
   def create(self,**kwargs):raise QuotaError('secret details must not be returned')
  c=MessageClassifierGPT(SimpleNamespace(chat=SimpleNamespace(completions=Provider())))
  result=c.classify('저축 계획','student')
  self.assertEqual(result.get('provider_error'),'insufficient_quota')
  self.assertNotIn('secret',str(result))
if __name__=='__main__':unittest.main()
