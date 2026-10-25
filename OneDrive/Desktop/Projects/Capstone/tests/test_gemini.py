import json
import unittest

from voiceops.chatbot import (
    GeminiAPIError,
    GeminiChatClient,
    GeminiResponseError,
    _classify_gemini_error,
    _parse_result,
)
from voiceops.config import Settings
from pathlib import Path
import os


class FakeModels:
    def __init__(self, responses=None, error=None):
        self.calls = []
        self.responses = list(responses or [])
        self.error = error

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.responses.pop(0)


class FakeGeminiClient:
    def __init__(self, models):
        self.models = models


class Response:
    def __init__(self, text):
        self.text = text


class SDKError(Exception):
    def __init__(self, code, status, message):
        super().__init__(message)
        self.code = code
        self.response_json = {
            "error": {"code": code, "status": status, "message": message}
        }


class GeminiAdapterTests(unittest.TestCase):
    def test_maps_messages_and_json_config(self):
        models = FakeModels([Response(json.dumps({"decision": "answer", "answer": "ok"}))])
        client = GeminiChatClient("test-key", client=FakeGeminiClient(models))
        response = client.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
        )
        self.assertEqual(response.text, '{"decision": "answer", "answer": "ok"}')
        self.assertEqual(models.calls[0]["model"], "gemini-2.5-flash")
        self.assertEqual(models.calls[0]["config"]["response_mime_type"], "application/json")
        self.assertIn("USER:\nquestion", models.calls[0]["contents"])

    def test_retries_transient_error_then_succeeds(self):
        models = FakeModels()
        calls = {"count": 0}

        def generate(**kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("503 service unavailable")
            return Response("{}")

        models.generate_content = generate
        sleeps = []
        client = GeminiChatClient(
            "test-key",
            client=FakeGeminiClient(models),
            backoff_seconds=0.01,
            sleep=sleeps.append,
            random_uniform=lambda _low, _high: 0,
        )
        client.create(model="gemini-2.5-flash", messages=[])
        self.assertEqual(calls["count"], 2)
        self.assertEqual(sleeps, [0.01])

    def test_quota_exhaustion_is_explicit_after_retries(self):
        models = FakeModels(
            error=SDKError(429, "RESOURCE_EXHAUSTED", "Daily quota exhausted")
        )
        sleeps = []
        client = GeminiChatClient(
            "test-key",
            client=FakeGeminiClient(models),
            backoff_seconds=0,
            sleep=sleeps.append,
        )
        with self.assertRaises(GeminiAPIError) as context:
            client.create(model="gemini-2.5-flash", messages=[])
        self.assertEqual(context.exception.category, "quota-exhausted")
        self.assertEqual(context.exception.status_code, 429)
        self.assertEqual(context.exception.api_error_code, "RESOURCE_EXHAUSTED")
        self.assertEqual(context.exception.exception_class, "test_gemini.SDKError")
        self.assertEqual(len(sleeps), 0)

    def test_classifies_rate_limit_separately(self):
        error = _classify_gemini_error(
            SDKError(429, "RESOURCE_EXHAUSTED", "Requests per minute limit reached")
        )
        self.assertEqual(error.category, "rate-limit")
        self.assertTrue(error.retryable)

    def test_quota_message_without_daily_word_is_permanent(self):
        error = _classify_gemini_error(
            SDKError(429, "RESOURCE_EXHAUSTED", "quota exhausted")
        )
        self.assertEqual(error.category, "quota-exhausted")
        self.assertFalse(error.retryable)

    def test_classifies_timeout_and_service_failures(self):
        timeout = _classify_gemini_error(RuntimeError("request timed out"))
        timed_out_service = _classify_gemini_error(
            SDKError(503, "UNAVAILABLE", "request timed out")
        )
        service = _classify_gemini_error(SDKError(503, "UNAVAILABLE", "temporary failure"))
        self.assertEqual(timeout.category, "timeout")
        self.assertTrue(timeout.retryable)
        self.assertEqual(timed_out_service.category, "timeout")
        self.assertEqual(service.category, "service")
        self.assertTrue(service.retryable)

    def test_classifies_authentication_and_invalid_model(self):
        auth = _classify_gemini_error(SDKError(401, "UNAUTHENTICATED", "bad key"))
        model = _classify_gemini_error(SDKError(404, "NOT_FOUND", "model not found"))
        self.assertEqual(auth.category, "authentication")
        self.assertEqual(model.category, "invalid-model")
        self.assertFalse(auth.retryable)
        self.assertFalse(model.retryable)

    def test_redacts_key_like_values(self):
        error = _classify_gemini_error(RuntimeError("api_key=secret-value"))
        self.assertNotIn("secret-value", error.safe_message)

    def test_parser_rejects_malformed_contract(self):
        with self.assertRaises(GeminiResponseError):
            _parse_result('{"decision":"answer","answer":"ok","extra":"unexpected"}')

    def test_settings_accepts_configured_gemini_model_and_rejects_other_provider(self):
        workspace = Path(__file__).resolve().parents[1]
        previous = os.environ.get("GEMINI_MODEL")
        try:
            os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"
            self.assertEqual(Settings.from_environment(workspace).llm_model, "gemini-2.5-flash")
            os.environ["GEMINI_MODEL"] = "gpt-4o-mini"
            with self.assertRaises(ValueError):
                Settings.from_environment(workspace)
        finally:
            if previous is None:
                os.environ.pop("GEMINI_MODEL", None)
            else:
                os.environ["GEMINI_MODEL"] = previous


if __name__ == "__main__":
    unittest.main()
