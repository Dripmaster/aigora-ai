import os
os.environ['OPENAI_API_KEY']='test-not-a-real-key'
import unittest,json
from types import SimpleNamespace
from fastapi.testclient import TestClient
import app.main as service

class Provider:
 def __init__(self,payload):self.payload=payload;self.prompts=[]
 def create(self,**kwargs):
  self.prompts.append(kwargs['messages'])
  return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload,ensure_ascii=False)))])
def sdk(provider):return SimpleNamespace(chat=SimpleNamespace(completions=provider))

class FinancialAPITests(unittest.TestCase):
 def setUp(self):self.client=TestClient(service.app)
 def test_question_uses_only_selected_lesson_material(self):
  provider=Provider({'need_question':True,'question':'매달 저축할 수 있는 금액은 얼마인가요?'})
  service.question_generator2.client=sdk(provider)
  response=self.client.post('/question',json={'nickname':'검증용','discussion_topic':'역산하기','video_id':'financial_1','chat_history':[],'questionText':''})
  self.assertEqual(response.status_code,200,response.text)
  self.assertIn('매달',response.json()['question'])
  prompt=provider.prompts[0][1]['content']
  self.assertIn('[교육 PDF 29쪽]',prompt)
  self.assertNotIn('[교육 PDF 5쪽]',prompt)
  self.assertNotIn('갓 구운 빵',prompt)
 def test_video_remap_keeps_each_lessons_theory_and_uses_selected_video_only(self):
  cases=[('financial_1',29,'하늘','목표 정하기'),('financial_2',42,'수빈','현재 점검'),('financial_3',52,'준호','진단'),('financial_4',42,'나래','기초적 노후 생활')]
  for key,page,person,topic in cases:
   with self.subTest(video_id=key):
    provider=Provider({'need_question':False,'reason':'대기','question':''})
    service.question_generator2.client=sdk(provider)
    response=self.client.post('/question',json={'nickname':'검증용','discussion_topic':topic,'video_id':key,'chat_history':[]})
    self.assertEqual(response.status_code,200,response.text)
    prompt=provider.prompts[0][1]['content']
    self.assertIn(f'[교육 PDF {page}쪽]',prompt)
    video=prompt.split('[영상 내용]')[1].split('[현재 차시 자료')[0]
    self.assertIn(person,video)
    self.assertIn(topic,video)
    for other in ['하늘','수빈','준호','나래']:
     if other!=person:self.assertNotIn(other,video)

 def test_unknown_scenario_rejected(self):
  r=self.client.post('/qa',json={'nickname':'검증용','discussion_topic':'테스트','video_id':'financial_5','chat_history':[],'questionText':'설명'})
  self.assertEqual(r.status_code,404)
 def test_classification_contract_matches_web_axis_keys(self):
  service.message_classifier_gpt.client=sdk(Provider({'cj_values':{'금융이해':1,'위험인식':60,'계획성':80,'실천의지':30},'summary':'예산과 기간을 설명함'}))
  r=self.client.post('/classify-gpt',json={'text':'매달 10만원씩 모으겠습니다','user_id':'검증용','context':{'lesson_id':2}})
  self.assertEqual(r.status_code,200,r.text)
  self.assertEqual(r.json()['cj_values']['금융이해'],1)
  self.assertEqual(r.json()['primary_trait'],'계획성')
 def test_evaluation_failure_has_no_fabricated_strength(self):
  class Offline:
   def create(self,**kwargs):raise RuntimeError('offline')
  service.discussion_evaluator.client=sdk(Offline())
  r=self.client.post('/evaluate',json={'user_id':'검증용','user_messages':[{'text':'의견'}]})
  self.assertEqual(r.status_code,200,r.text)
  self.assertEqual(r.json()['strengths'],[])
  self.assertEqual(r.json()['evaluation_method'],'평가불가')
if __name__=='__main__':unittest.main()
