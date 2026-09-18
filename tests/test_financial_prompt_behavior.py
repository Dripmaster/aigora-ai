import os
os.environ['OPENAI_API_KEY']='test-not-a-real-key'
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from app.question_generator2 import QuestionGenerator2
from app.discussion_evaluator import PersonalEvaluator

class PromptBehaviorTests(unittest.TestCase):
 def test_invalid_question_output_is_not_broadcast_as_model_text(self):
  for content in ['임의의 지시문', '{"need_question":"false","question":"잘못된 질문"}', '{"need_question":true,"question":42}']:
   with self.subTest(content=content):
    provider=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))]))))
    generator=QuestionGenerator2();generator.client=provider
    self.assertEqual(generator.generate_question('학생','위험 신호','시나리오','자료',[]),'결과없음')
 def test_unavailable_question_generator_stays_quiet(self):
  generator=QuestionGenerator2();generator.gpt_enabled=False
  self.assertEqual(generator.generate_question('학생','위험 신호','시나리오','자료',[]),'결과없음')
 def test_no_record_does_not_infer_personality_or_lack_of_effort(self):
  result=PersonalEvaluator().evaluate_user('학생',[])
  self.assertEqual(result['improvements'],[])
  self.assertIn('기록',result['participation_summary'])
  self.assertNotIn('기회를 놓치',result['personalized_feedback'])

 def test_unrelated_message_ids_cannot_retain_high_relevance(self):
  from app.discussion_summarizer import DiscussionSummarizer
  summary=DiscussionSummarizer()
  result=summary._resolve_result({'user_id':'학생','topics':[{'topic':'목표','relevance_score':.95,'related_message_ids':[999],'summary':'근거 없는 요약'}]},[{'id':1,'text':'내 의견'}],[{'name':'목표'},{'name':'실행'}])
  for topic in result['topics']:
   self.assertEqual(topic['related_statements'],[])
   self.assertEqual(topic['relevance_score'],0)
   self.assertEqual(topic['summary'],'')
 def test_overall_provider_receives_actual_axis_definitions(self):
  from app.financial_profile import DEFINITIONS
  captured=[]
  def create(**kwargs):
   captured.append(kwargs['messages'][0]['content'])
   return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"discussion_summary":"발언 요약"}'))])
  evaluator=PersonalEvaluator();evaluator.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  evaluator.evaluate_discussion_overall([{'nickname':'학생','text':'의견'}])
  for name,definition in DEFINITIONS.items():
   self.assertIn(name,captured[0]);self.assertIn(definition,captured[0])
