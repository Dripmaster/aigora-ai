import unittest
from app.financial_profile import normalized_scores, lesson_text, unassessed_result

class FinancialContractTests(unittest.TestCase):
 def test_only_financial_axes_survive_and_scores_remain_0_to_100(self):
  self.assertEqual(normalized_scores({'금융이해':1,'위험인식':140,'계획성':-4,'실천의지':'bad','정직':100}), {'금융이해':1,'위험인식':100,'계획성':0,'실천의지':0})
 def test_unknown_lesson_never_receives_all_other_lessons(self):
  source={'video_content':{'financial_2':{'lesson_id':2}},'slide_content':{'lesson_1':'사기 원문','lesson_2':'저축 원문'}}
  self.assertEqual(lesson_text(source,'financial_2'),'저축 원문')
  with self.assertRaises(ValueError): lesson_text(source,'financial_5')
 def test_failure_does_not_award_traits_from_generic_keywords(self):
  result=unassessed_result()
  self.assertEqual(sum(result['cj_values'].values()),0)
  self.assertEqual(result['primary_trait'],'평가불가')
  self.assertEqual(result['evaluation_status'],'unavailable')
if __name__=='__main__':unittest.main()
