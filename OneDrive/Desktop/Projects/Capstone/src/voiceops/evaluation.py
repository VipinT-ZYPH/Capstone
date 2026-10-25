"""Dataset-driven evaluation for prompt-only chatbot answers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
import time
from typing import Any, Iterable

from .chatbot import ChatResult, ZentroMartChatbot
from .rag_chatbot import RAGChatbot, evidence_as_dicts


@dataclass(frozen=True)
class EvaluationRecord:
    question_id: str
    question: str
    answer: str
    decision: str
    expected_decision: str
    reference_answer: str
    answerable: str
    decision_correct: bool
    required_facts_found: bool
    forbidden_claims_absent: bool
    correct: bool
    scored: bool = True
    error: str = ""
    retrieved_evidence: list[dict[str, object]] = field(default_factory=list)
    evidence_recall: bool | None = None
    error_type: str = ""
    error_status_code: int | None = None
    error_api_code: str | int | None = None
    error_message: str = ""
    error_category: str = ""


def evaluate_questions(
    questions: Iterable[dict[str, str]],
    chatbot: ZentroMartChatbot,
    *,
    request_delay_seconds: float = 0.0,
    sleep: Any = time.sleep,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for index, question in enumerate(questions):
        if index and request_delay_seconds:
            sleep(request_delay_seconds)
        try:
            result = chatbot.answer(question["question"])
            records.append(_score(question, result))
        except Exception as exc:
            records.append(
                EvaluationRecord(
                    question_id=question["id"],
                    question=question["question"],
                    answer="",
                    decision="error",
                    expected_decision=question.get("expected_decision", ""),
                    reference_answer=question.get("reference_answer", ""),
                    answerable=question.get("answerable", ""),
                    decision_correct=False,
                    required_facts_found=False,
                    forbidden_claims_absent=False,
                    correct=False,
                    scored=False,
                    error=str(exc),
                    **_error_fields(exc),
                )
            )
    return records


def evaluate_rag_questions(
    questions: Iterable[dict[str, str]],
    chatbot: RAGChatbot,
    *,
    request_delay_seconds: float = 0.0,
    sleep: Any = time.sleep,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for index, question in enumerate(questions):
        if index and request_delay_seconds:
            sleep(request_delay_seconds)
        try:
            result, evidence = chatbot.answer(question["question"])
            base = _score(question, result)
            gold_ids = {
                item.strip() for item in question.get("gold_evidence", "").split(" ; ")
                if item.strip() and not item.startswith("products.csv:")
            }
            active_ids = {item.section_id for item in chatbot.index.sections}
            gold_ids.intersection_update(active_ids)
            retrieved_ids = {item.section_id for item in evidence}
            recall = gold_ids.issubset(retrieved_ids) if gold_ids else True
            records.append(
                replace(
                    base,
                    retrieved_evidence=evidence_as_dicts(evidence),
                    evidence_recall=recall,
                )
            )
        except Exception as exc:
            records.append(
                EvaluationRecord(
                    question_id=question["id"],
                    question=question["question"],
                    answer="",
                    decision="error",
                    expected_decision=question.get("expected_decision", ""),
                    reference_answer=question.get("reference_answer", ""),
                    answerable=question.get("answerable", ""),
                    decision_correct=False,
                    required_facts_found=False,
                    forbidden_claims_absent=False,
                    correct=False,
                    scored=False,
                    error=str(exc),
                    evidence_recall=False,
                    **_error_fields(exc),
                )
            )
    return records


def _error_fields(exc: Exception) -> dict[str, object]:
    return {
        "error_type": getattr(exc, "exception_class", f"{type(exc).__module__}.{type(exc).__name__}"),
        "error_status_code": getattr(exc, "status_code", None),
        "error_api_code": getattr(exc, "api_error_code", None),
        "error_message": getattr(exc, "safe_message", str(exc)),
        "error_category": _error_category(exc),
    }


def _error_category(exc: Exception) -> str:
    category = getattr(exc, "category", "")
    if category:
        return str(category)
    return "application"


def records_as_dicts(records: Iterable[EvaluationRecord]) -> list[dict[str, object]]:
    return [asdict(record) for record in records]


def _score(question: dict[str, str], result: ChatResult) -> EvaluationRecord:
    answer = result.answer.casefold()
    required = _all_alternatives_present(question.get("must_include", ""), answer)
    forbidden = not any(
        phrase.strip().casefold() in answer
        for phrase in question.get("must_not_include", "").split(" ; ")
        if phrase.strip()
    )
    expected_decision = question.get("expected_decision", "")
    acceptable_decisions = {
        item.strip()
        for item in question.get("acceptable_decisions", expected_decision).split(" ; ")
        if item.strip()
    }
    if not acceptable_decisions:
        acceptable_decisions = {expected_decision}
    decision_correct = result.decision in acceptable_decisions
    return EvaluationRecord(
        question_id=question["id"],
        question=question["question"],
        answer=result.answer,
        decision=result.decision,
        expected_decision=expected_decision,
        reference_answer=question.get("reference_answer", ""),
        answerable=question.get("answerable", ""),
        decision_correct=decision_correct,
        required_facts_found=required,
        forbidden_claims_absent=forbidden,
        correct=decision_correct and required and forbidden,
    )


def _all_alternatives_present(requirements: str, answer: str) -> bool:
    for requirement in requirements.split(" ; "):
        alternatives = [part.strip() for part in requirement.split(" / ") if part.strip()]
        if alternatives and not any(_normalise(part) in _normalise(answer) for part in alternatives):
            return False
    return True


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()
