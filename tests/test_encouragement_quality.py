"""Regression coverage for the CJ encouragement variety contract, without API calls."""
import random
import unittest
from unittest.mock import patch

from app.participant_monitor import ParticipantMonitor


class EncouragementQualityTests(unittest.TestCase):
    def test_fallback_does_not_repeat_within_six_consecutive_invitations(self):
        for level in (1, 2, 3):
            with self.subTest(level=level):
                monitor = ParticipantMonitor.__new__(ParticipantMonitor)
                monitor.gpt_enabled = False
                monitor.recent_messages = []
                rng = random.Random(7)
                with patch('app.participant_monitor.random.choice', side_effect=rng.choice):
                    messages = [monitor.generate_encouragement_message('학생', [], level)
                                for _ in range(6)]
                self.assertEqual(len(set(messages)), 6)
                self.assertTrue(all(message.startswith('학생님') for message in messages))


if __name__ == '__main__':
    unittest.main()
