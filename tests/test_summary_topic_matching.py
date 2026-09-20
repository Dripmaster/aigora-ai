import unittest
from app.discussion_summarizer import DiscussionSummarizer


class SummaryTopicMatchingTests(unittest.TestCase):
    def setUp(self):
        self.summarizer = DiscussionSummarizer.__new__(DiscussionSummarizer)
        self.messages = [{'id': 1, 'text': '노트북을 사려고 1년 동안 120만원을 모으겠습니다.'}]

    def resolve(self, names, returned, ids=None):
        return self.summarizer._resolve_result(
            {'user_id': 'student', 'topics': [{
                'topic': returned, 'relevance_score': 0.9,
                'related_message_ids': [1] if ids is None else ids,
                'summary': '노트북 구입을 위해 1년간 120만원을 모을 계획이라고 설명했다.',
            }]},
            self.messages, [{'name': name} for name in names],
        )['topics']

    def test_short_title_keeps_generated_summary_and_original_question(self):
        name = '목표 정하기: 1년 안에 이루고 싶은 금전적 목표는 무엇인가요?'
        topic = self.resolve([name], '목표 정하기')[0]
        self.assertEqual(topic['topic'], name)
        self.assertEqual(topic['related_statements'], [self.messages[0]['text']])
        self.assertEqual(topic['summary'], '노트북 구입을 위해 1년간 120만원을 모을 계획이라고 설명했다.')

    def test_shared_short_title_does_not_assign_summary_to_two_questions(self):
        names = ['목표 정하기: 단기 목표는?', '목표 정하기: 장기 목표는?']
        self.assertEqual([t['summary'] for t in self.resolve(names, '목표 정하기')], ['', ''])
        exact = self.resolve(names, names[1])
        self.assertEqual(exact[0]['summary'], '')
        self.assertTrue(exact[1]['summary'])

    def test_short_title_cannot_steal_another_exact_topic(self):
        topics = self.resolve(['목표 정하기', '목표 정하기: 구체적 금액은?'], '목표 정하기')
        self.assertTrue(topics[0]['summary'])
        self.assertEqual(topics[1]['summary'], '')

    def test_unknown_title_is_not_matched_by_partial_words(self):
        for returned in ['목표', '다른 목표 정하기', '목표 정하기: 관계없는 질문']:
            with self.subTest(returned=returned):
                self.assertEqual(self.resolve(['목표 정하기: 금액은?'], returned)[0]['summary'], '')

    def test_short_title_still_requires_target_speakers_message_evidence(self):
        topic = self.resolve(['목표 정하기: 금액은?'], '목표 정하기', ids=[999])[0]
        self.assertEqual(topic['summary'], '')
        self.assertEqual(topic['related_statements'], [])
        self.assertEqual(topic['relevance_score'], 0)

    def test_existing_exact_and_description_suffix_matches_are_preserved(self):
        for returned in ['목표 정하기', '목표 정하기 - 저축 계획']:
            with self.subTest(returned=returned):
                self.assertTrue(self.resolve(['목표 정하기'], returned)[0]['summary'])
