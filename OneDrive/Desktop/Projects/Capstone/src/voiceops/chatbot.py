"""Prompt-only ZentroMart chatbot backed by Google's Gemini API."""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

SYSTEM_PROMPT = """You are ZentroMart's customer support agent.
Answer the customer's message clearly and courteously using only the information
you know from the conversation and the supplied synthetic ZentroMart context.
This is prompt-only Step 1: do not claim to have searched a knowledge base or
used tools, and do not invent policies, prices, dates, order details, or
eligibility. If the information is not available, say that you cannot confirm
it and offer to clarify or escalate. Never reveal private order information
without caller verification; for an order-specific request, ask for the order
ID and either the registered phone last four digits or delivery pincode.
Treat 20 October 2026 as today. Respond in the customer's language when
possible. Return valid JSON with exactly two string fields:
{"decision":"answer|clarify|escalate|refuse","answer":"..."}.
"""


class ChatCompletionClient(Protocol):
    def create(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class ChatResult:
    answer: str
    decision: str
    raw: str


class GeminiAPIError(RuntimeError):
    """A Gemini request failed without exposing credentials."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        retryable: bool,
        exception_class: str = "",
        status_code: int | None = None,
        api_error_code: str | int | None = None,
        original_message: str = "",
    ):
        safe_message = _safe_error_message(message)
        super().__init__(f"Gemini {category} error: {safe_message}")
        self.category = category
        self.retryable = retryable
        self.exception_class = exception_class
        self.status_code = status_code
        self.api_error_code = api_error_code
        self.safe_message = safe_message
        self.original_message = _safe_error_message(original_message or message)


class GeminiResponseError(RuntimeError):
    """A model response could not be parsed as the required contract."""

    category = "parsing"
    retryable = False
    exception_class = ""
    status_code = None
    api_error_code = None

    def __init__(self, message: str):
        safe_message = _safe_error_message(message)
        super().__init__(safe_message)
        self.safe_message = safe_message


class GeminiChatClient:
    """Adapter from the project's message contract to ``google-genai``."""

    def __init__(
        self,
        api_key: str,
        *,
        client: Any | None = None,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        sleep: Any = time.sleep,
        random_uniform: Any = random.uniform,
    ):
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY must not be empty")
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "The google-genai package is required. Install requirements.txt first."
                ) from exc
            client = genai.Client(api_key=api_key)
        self._client = client
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._random_uniform = random_uniform

    def create(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        system_instruction = next(
            (message["content"] for message in messages if message["role"] == "system"),
            "",
        )
        contents = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}"
            for message in messages
            if message["role"] != "system"
        )
        config = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "temperature": 0,
        }
        config.update(kwargs.pop("config", {}))
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                error = _classify_gemini_error(exc)
                if not error.retryable or attempt == self.max_retries:
                    raise error from exc
                base = self.backoff_seconds * (2**attempt)
                self._sleep(base + self._random_uniform(0, base * 0.25))


class ZentroMartChatbot:
    def __init__(self, client: ChatCompletionClient, model: str):
        self.client = client
        self.model = model
        self.history: list[dict[str, str]] = []

    def answer(self, question: str) -> ChatResult:
        if not question.strip():
            raise ValueError("question must not be empty")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history]
        messages.append({"role": "user", "content": question})
        response = self.client.create(model=self.model, messages=messages)
        raw = _response_text(response)
        result = _parse_result(raw)
        self.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": raw},
            ]
        )
        return result

    def reset(self) -> None:
        self.history.clear()


def _response_text(response: Any) -> str:
    try:
        content = response.text
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise GeminiResponseError("Gemini response did not contain text") from exc
    if not isinstance(content, str) or not content.strip():
        raise GeminiResponseError("LLM response contained empty content")
    return content


def _parse_result(raw: str) -> ChatResult:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiResponseError("LLM returned invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"decision", "answer"}:
        raise GeminiResponseError(
            'LLM JSON response must contain exactly "decision" and "answer" fields'
        )
    answer = payload.get("answer")
    decision = payload.get("decision")
    if not isinstance(answer, str) or not answer.strip():
        raise GeminiResponseError("LLM JSON response is missing a non-empty answer")
    if decision not in {"answer", "clarify", "escalate", "refuse"}:
        raise GeminiResponseError("LLM JSON response has an invalid decision")
    return ChatResult(answer=answer.strip(), decision=decision, raw=raw)


def _classify_gemini_error(exc: Exception) -> GeminiAPIError:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    response_json = getattr(exc, "response_json", None)
    error_payload = response_json.get("error", {}) if isinstance(response_json, dict) else {}
    api_error_code = error_payload.get("status") or error_payload.get("code")
    detail = error_payload.get("message") if isinstance(error_payload, dict) else None
    original_message = str(detail or exc)
    text = original_message.casefold()
    exception_class = f"{type(exc).__module__}.{type(exc).__name__}"
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    common = {
        "exception_class": exception_class,
        "status_code": status,
        "api_error_code": api_error_code,
        "original_message": original_message,
    }
    if status in {401, 403} or "api key" in text or "permission" in text or "unauthenticated" in text:
        return GeminiAPIError("authentication", "check GEMINI_API_KEY", retryable=False, **common)
    if (
        "invalid model" in text
        or "model not found" in text
        or "not found" in text and "model" in text
    ):
        return GeminiAPIError("invalid-model", "check GEMINI_MODEL", retryable=False, **common)
    if status == 429 or "resource exhausted" in text or "quota" in text:
        quota = any(
            term in text
            for term in (
                "daily",
                "per day",
                "quota exceeded",
                "quota exhausted",
                "quota limit",
                "insufficient_quota",
                "credits remaining",
                "billing",
            )
        )
        if quota:
            return GeminiAPIError(
                "quota-exhausted",
                "Gemini quota is exhausted",
                retryable=False,
                **common,
            )
        return GeminiAPIError(
            "rate-limit",
            "Gemini request rate limit reached",
            retryable=True,
            **common,
        )
    if status == 408 or "timeout" in text or "timed out" in text or "deadline exceeded" in text:
        return GeminiAPIError("timeout", "request timed out", retryable=True, **common)
    if status is not None:
        try:
            if int(status) >= 500:
                return GeminiAPIError("service", "Gemini service unavailable", retryable=True, **common)
        except (TypeError, ValueError):
            pass
    if "service unavailable" in text or "temporarily unavailable" in text or any(
        f"{code}" in text for code in (500, 502, 503, 504)
    ):
        return GeminiAPIError("service", "Gemini service unavailable", retryable=True, **common)
    return GeminiAPIError("request", "request could not be completed", retryable=False, **common)


def _safe_error_message(message: str) -> str:
    """Remove credential-like values before an error reaches records or logs."""
    return re.sub(r"(?i)(api[_ -]?key|key|token|authorization)\s*[:=]\s*\S+", r"\1=[REDACTED]", message)
