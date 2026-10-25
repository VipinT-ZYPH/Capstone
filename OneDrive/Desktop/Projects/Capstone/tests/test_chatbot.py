import json
import unittest

from voiceops.chatbot import GeminiAPIError, ZentroMartChatbot
from voiceops.evaluation import evaluate_questions


class FakeResponse:
    def __init__(self, content):
        self.text = content


class FakeClient:
    def __init__(self, decision="answer", answer="12 months via My Orders"):
        self.calls = []
        self.decision = decision
        self.answer_text = answer

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(json.dumps({"decision": self.decision, "answer": self.answer_text}))


class ChatbotTests(unittest.TestCase):
    def test_history_is_sent_and_reset(self):
        client = FakeClient()
        bot = ZentroMartChatbot(client, "test-model")
        bot.answer("What is the warranty?")
        bot.answer("How do I claim it?")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(client.calls[1]["messages"]), 4)
        bot.reset()
        self.assertEqual(bot.history, [])

    def test_evaluation_uses_dataset_expectations(self):
        bot = ZentroMartChatbot(FakeClient(), "test-model")
        records = evaluate_questions(
            [
                {
                    "id": "Q1",
                    "question": "Warranty?",
                    "expected_decision": "answer",
                    "must_include": "12 ; My Orders / app",
                    "must_not_include": "no warranty",
                }
            ],
            bot,
        )
        self.assertTrue(records[0].correct)
        self.assertTrue(records[0].decision_correct)

    def test_acceptable_decision_is_used_when_provided(self):
        bot = ZentroMartChatbot(
            FakeClient(decision="escalate", answer="I can connect you to a colleague."),
            "test-model",
        )
        records = evaluate_questions(
            [
                {
                    "id": "Q1",
                    "question": "Unsupported?",
                    "expected_decision": "answer",
                    "acceptable_decisions": "answer ; escalate",
                }
            ],
            bot,
        )
        self.assertTrue(records[0].decision_correct)
        self.assertTrue(records[0].correct)

    def test_invalid_llm_json_is_reported_as_evaluation_error(self):
        class InvalidClient:
            def create(self, **kwargs):
                return FakeResponse("not json")

        records = evaluate_questions(
            [{"id": "Q1", "question": "Hi", "expected_decision": "answer"}],
            ZentroMartChatbot(InvalidClient(), "test-model"),
        )
        self.assertEqual(records[0].decision, "error")
        self.assertFalse(records[0].correct)
        self.assertFalse(records[0].scored)

    def test_evaluation_paces_questions_and_preserves_unicode(self):
        sleeps = []
        bot = ZentroMartChatbot(FakeClient(answer="हिंदी सहायता"), "test-model")
        questions = [
            {"id": "Q1", "question": "हिंदी?", "expected_decision": "answer"},
            {"id": "Q2", "question": "Hinglish?", "expected_decision": "answer"},
            {"id": "Q3", "question": "தமிழ்?", "expected_decision": "answer"},
        ]
        records = evaluate_questions(
            questions,
            bot,
            request_delay_seconds=0.25,
            sleep=sleeps.append,
        )
        self.assertEqual(sleeps, [0.25, 0.25])
        self.assertEqual(records[0].answer, "हिंदी सहायता")

    def test_evaluation_keeps_api_failure_unscored_and_structured(self):
        class QuotaClient:
            def create(self, **kwargs):
                raise GeminiAPIError(
                    "quota-exhausted",
                    "Gemini quota is exhausted",
                    retryable=False,
                    status_code=429,
                )

        records = evaluate_questions(
            [{"id": "Q1", "question": "Hi", "expected_decision": "answer"}],
            ZentroMartChatbot(QuotaClient(), "test-model"),
        )
        self.assertFalse(records[0].scored)
        self.assertFalse(records[0].correct)
        self.assertEqual(records[0].error_category, "quota-exhausted")
        self.assertEqual(records[0].error_status_code, 429)


if __name__ == "__main__":
    unittest.main()
