import os
os.environ['OPENAI_API_KEY']='test-not-a-real-key'
import asyncio,json,threading,unittest
from unittest.mock import patch
from types import SimpleNamespace
import httpx
from fastapi.testclient import TestClient
import app.main as service
from app.discussion_summarizer import DiscussionSummarizer

class FinancialAuditTests(unittest.TestCase):
 def setUp(self): self.client=TestClient(service.app)
 def test_empty_classification_is_client_error(self):
  for path in ['/classify','/classify-gpt']:
   with self.subTest(path=path):
    self.assertEqual(self.client.post(path,json={'user_id':'student','text':'  '}).status_code,400)
 def test_failed_answer_does_not_endorse_question(self):
  with patch.object(service.answer_generator,'gpt_enabled',False):
   r=self.client.post('/qa',json={'nickname':'student','discussion_topic':'대응 정하기','video_id':'financial_1','questionText':'모르는 사람에게 인증번호를 보내도 되나요?','chat_history':[]})
   self.assertEqual(r.status_code,200)
   self.assertIn('답변을 생성하지 못',r.json()['question'])
 def test_question_does_not_require_unused_question_text(self):
  with patch.object(service.question_generator2,'generate_question',return_value='결과없음'):
   r=self.client.post('/question',json={'nickname':'student','discussion_topic':'역산하기','video_id':'financial_2','chat_history':[]})
   self.assertEqual(r.status_code,200,r.text)
 def test_qa_rejects_empty_question(self):
  with patch.object(service.answer_generator,'generate_answer',return_value='unexpected'):
   self.assertEqual(self.client.post('/qa',json={'nickname':'student','discussion_topic':'역산하기','video_id':'financial_2','questionText':' ','chat_history':[]}).status_code,400)
 def test_summary_retains_early_target_speaker(self):
  summary=DiscussionSummarizer()
  history=[{'nickname':'early','text':'월급날 자동이체를 하겠습니다.'}]+[{'nickname':'later','text':f'발언{i}'} for i in range(70)]
  with patch.object(summary,'_analyze_with_gpt',return_value={'user_id':'early','topics':[]}) as provider:
   summary.summarize_user('early',history,[{'name':'실행 문장'}])
   provider.assert_called_once()
   self.assertEqual(provider.call_args.args[2][0]['text'],history[0]['text'])
 def test_qa_uses_fixed_lesson_material_without_query_expansion(self):
  with patch.object(service.answer_generator,'generate_answer',return_value='답변') as answer:
   prompts=[]
   for question in ['정기적금과 자유적금의 차이는?', '복리와 대출은 무엇인가요?']:
    r=self.client.post('/qa',json={'nickname':'student','discussion_topic':'상품 고르기','video_id':'financial_2','questionText':question,'chat_history':[]})
    self.assertEqual(r.status_code,200)
    prompts.append(answer.call_args.kwargs['slide_content'])
   self.assertIn('[교육 PDF',prompts[0])
   self.assertEqual(prompts[0],prompts[1])
   self.assertNotIn('[보충 참고자료',prompts[0])
 def test_classifier_keeps_lesson_context_without_reference_injection(self):
  from app.financial_profile import unassessed_result
  with patch.object(service.message_classifier_gpt,'classify',return_value=unassessed_result()) as classifier:
   for path in ['/classify-gpt']:
    r=self.client.post(path,json={'text':'보험료가 부담되므로 보장 범위를 확인하겠습니다.','user_id':'student','context':{'lesson_id':3,'discussion_topic':'현재 점검','reference_material':'외부 자료'}})
    self.assertEqual(r.status_code,200)
    self.assertEqual(classifier.call_args.args[2],{'lesson_id':3,'discussion_topic':'현재 점검'})
 def test_form_uses_current_contract_and_escapes_input(self):
  from app.financial_profile import unassessed_result
  with patch.object(service.message_classifier,'classify',return_value=unassessed_result()):
   r=self.client.post('/form/result',data={'user_id':'<script>x</script>','text':'의견'})
   self.assertNotIn('오류 발생',r.text)
   self.assertNotIn('<script>',r.text)
   self.assertIn('평가불가',r.text)

class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
 async def test_slow_provider_does_not_block_other_requests(self):
  finished=threading.Event();release=threading.Event()
  def slow(*args,**kwargs):
   release.wait(0.6);finished.set()
   from app.financial_profile import unassessed_result
   return unassessed_result()
  with patch.object(service.message_classifier_gpt,'classify',side_effect=slow):
   async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app),base_url='http://test') as client:
    task=asyncio.create_task(client.post('/classify-gpt',json={'text':'의견','user_id':'test'}))
    try:
     await asyncio.sleep(0.03)
     self.assertEqual((await client.get('/')).status_code,200)
     self.assertFalse(finished.is_set(),'provider blocked the event loop')
    finally:
     release.set();await task
