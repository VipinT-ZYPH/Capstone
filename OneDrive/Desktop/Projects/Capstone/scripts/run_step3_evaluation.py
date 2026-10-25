"""Run the deterministic offline Step 3 conversation evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from voiceops.chatbot import ChatResult
from voiceops.conversation import ConversationalRAGAgent
from voiceops.conversation_evaluation import score_conversations, write_report
from voiceops.dataset import DatasetPaths, DatasetRepository
from voiceops.rag import KnowledgeBaseIndex
from voiceops.rag_chatbot import RAGChatbot


class MockResponse:
    def __init__(self, payload: dict[str, str]):
        self.text = json.dumps(payload)


class MockClient:
    """Deterministic turn-aware mock; not a live accuracy test."""

    def __init__(self, scenario_by_order: dict[str, str] | None = None):
        self.scenario_by_order = scenario_by_order or {}

    def create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        question = _current_customer_question(prompt)
        question_lower = question.casefold()
        evidence_ids = _evidence_ids(prompt)
        verified_context = "Verified order context:" in prompt
        scenario = _scenario_from_prompt(prompt, self.scenario_by_order)

        if _requests_human(question_lower) or _requires_handover(question_lower, scenario):
            return MockResponse(
                {
                    "decision": "escalate",
                    "answer": "I will hand this over to a human colleague for review.",
                }
            )
        if not evidence_ids:
            return MockResponse(
                {
                    "decision": "clarify",
                    "answer": "I could not find enough policy information to answer this safely.",
                }
            )
        if _is_protected_order_request(question_lower) and not verified_context:
            return MockResponse(
                {
                    "decision": "clarify",
                    "answer": "Please verify the order before I discuss order-specific details.",
                }
            )

        grounding = ", ".join(evidence_ids[:3])
        context_note = " using the verified order context" if verified_context else ""
        return MockResponse(
            {
                "decision": "answer",
                "answer": (
                    f"Based on the retrieved ZentroMart sections {grounding}, "
                    f"I can provide guidance{context_note}. Please follow the "
                    "current policy described there."
                ),
            }
        )


def _current_customer_question(prompt: str) -> str:
    marker = "Customer question:\n"
    if marker in prompt:
        question = prompt.split(marker, 1)[1]
        return question.split("\n\nRetrieved evidence:", 1)[0].strip()
    marker = "Customer message:\n"
    if marker in prompt:
        question = prompt.split(marker, 1)[1]
        return question.split("\n\nVerified order context:", 1)[0].strip()
    return prompt.strip()


def _evidence_ids(prompt: str) -> list[str]:
    evidence_marker = "Retrieved evidence:"
    if evidence_marker not in prompt:
        return []
    evidence = prompt.split(evidence_marker, 1)[1]
    return [
        match
        for match in re.findall(r"\[([A-Za-z0-9][A-Za-z0-9.-]*)\]", evidence)
        if match not in {"REDACTED"}
    ]


def _requests_human(question: str) -> bool:
    return any(
        phrase in question
        for phrase in (
            "speak to a person",
            "talk to a person",
            "human agent",
            "human colleague",
            "insaan se",
            "किसी इंसान",
            "मानव एजेंट",
        )
    )


def _requires_handover(question: str, scenario: str = "") -> bool:
    return any(
        phrase in question
        for phrase in (
            "third time",
            "third contact",
            "teesri baar",
            "तीसरी बार",
            "police complaint",
            "consumer court",
            "legal action",
            "replacement approve",
            "replacement approval",
        )
    )


def _scenario_from_prompt(prompt: str, scenario_by_order: dict[str, str]) -> str:
    for order_id in re.findall(r"(?<!\d)\d{8}(?!\d)", prompt):
        if order_id in scenario_by_order:
            return scenario_by_order[order_id]
    return ""


def _is_protected_order_request(question: str) -> bool:
    return any(
        phrase in question
        for phrase in (
            "where is my order",
            "order status",
            "refund",
            "रिफ़ंड",
            "order kab",
            "मेरा ऑर्डर",
        )
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repository = DatasetRepository(DatasetPaths.discover(project_root))
    index = KnowledgeBaseIndex(repository.sections())
    customers = {
        row["customer_id"]: row
        for row in repository.read_csv("data/customers.csv")
    }
    valid_values = {}
    for row in repository.read_csv("data/orders.csv"):
        customer = customers[row["customer_id"]]
        valid_values[row["order_id"]] = {
            customer["phone"].replace(" ", "")[-4:],
            row["delivery_pincode"],
        }

    def lookup_order(order_id: str, verification_value: str | None = None):
        if verification_value is None:
            return {"verification_required": True, "order_id": order_id}
        verified = verification_value in valid_values.get(order_id, set())
        return (
            {
                "found": True,
                "verified": True,
                "order_id": order_id,
                "status": "delivered",
                "contacts_last_7_days": 2 if order_id in {"45011817", "48196636"} else 0,
            }
            if verified
            else {"verified": False, "verification_required": True}
        )

    def create_ticket(summary: str, **kwargs):
        return {"created": True, "ticket_id": "TKT-OFFLINE"}

    rows = repository.practice_conversations()
    scenario_by_order = {
        row["order_id"]: row["scenario"]
        for row in rows
    }
    agent = ConversationalRAGAgent(
        lambda: RAGChatbot(MockClient(scenario_by_order), "offline-mock", index),
        lookup_order=lookup_order,
        create_ticket=create_ticket,
    )
    scores = score_conversations(rows, agent)
    per_turn, aggregate = write_report(project_root / "outputs" / "step3", rows, scores)
    print(json.dumps({"per_turn": str(per_turn), "aggregate": str(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
