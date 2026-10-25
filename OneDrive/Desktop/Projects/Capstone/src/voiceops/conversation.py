"""Multi-turn conversational orchestration for the ZentroMart RAG agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from .chatbot import ChatResult
from .rag_chatbot import RAGChatbot, evidence_as_dicts


class OrderLookup(Protocol):
    def __call__(self, order_id: str, verification_value: str | None = None) -> dict[str, Any]:
        ...


class TicketCreator(Protocol):
    def __call__(
        self,
        summary: str,
        order_id: str | None = None,
        reason: str | None = None,
        intent: str | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass
class ConversationState:
    conversation_id: str
    order_id: str | None = None
    issue: str = ""
    verified: bool = False
    verification_attempts: int = 0
    order: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    ticket_id: str | None = None
    ticket_attempted: bool = False
    turns: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationTurnResult:
    conversation_id: str
    result: ChatResult
    evidence: list[dict[str, object]]
    tool_calls: list[dict[str, object]]
    evidence_applicable: bool = True
    handover_summary: str = ""
    scored: bool = True
    error_category: str = ""
    error_message: str = ""


class ConversationalRAGAgent:
    """Keep conversation state isolated while reusing the RAG answer flow."""

    def __init__(
        self,
        rag_factory: Callable[[], RAGChatbot],
        *,
        lookup_order: OrderLookup | None = None,
        create_ticket: TicketCreator | None = None,
    ):
        self._rag_factory = rag_factory
        self._lookup_order = lookup_order or _load_lookup_order()
        self._create_ticket = create_ticket or _load_create_ticket()
        self._conversations: dict[str, tuple[ConversationState, RAGChatbot]] = {}

    def process_turn(self, conversation_id: str, customer_message: str) -> ConversationTurnResult:
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        if not customer_message.strip():
            raise ValueError("customer_message must not be empty")
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            conversation = (
                ConversationState(conversation_id=conversation_id),
                self._rag_factory(),
            )
            self._conversations[conversation_id] = conversation
        state, rag = conversation
        state.turns.append({"role": "user", "content": customer_message})
        state.issue = _merge_issue(state.issue, customer_message)
        order_id = _extract_order_id(customer_message)
        if order_id:
            if state.order_id and state.order_id != order_id:
                state.order = {}
                state.verified = False
                state.verification_attempts = 0
                state.actions.append("cleared previous order context")
            state.order_id = order_id

        tool_calls: list[dict[str, object]] = []
        if not state.order_id and _needs_order_context(customer_message):
            try:
                generated, evidence = rag.answer(customer_message)
            except Exception as exc:
                return self._error_turn(state, [], [], exc)
            result = ChatResult(
                answer="I’m an AI assistant. Please share your order ID so I can help with the order.",
                decision="clarify",
                raw=generated.raw,
            )
            state.actions.append("requested order ID")
            return self._finish(state, result, evidence, tool_calls)

        if state.order_id and not state.verified:
            verification = _extract_verification_value(customer_message, state.order_id)
            if verification is None:
                try:
                    lookup = self._lookup_order(state.order_id)
                except Exception as exc:
                    return self._error_turn(
                        state,
                        [{"tool": "lookup_order", "arguments": {"order_id": state.order_id}}],
                        [],
                        exc,
                    )
                tool_calls.append({"tool": "lookup_order", "arguments": {"order_id": state.order_id}, "result": lookup})
                state.actions.append("requested identity verification")
                return self._finish(
                    state,
                    ChatResult(
                        answer="Please provide either the last four digits of the registered phone number or the delivery pincode.",
                        decision="clarify",
                        raw="",
                    ),
                    [],
                    tool_calls,
                    evidence_applicable=False,
                )
            state.verification_attempts += 1
            try:
                lookup = self._lookup_order(state.order_id, verification)
            except Exception as exc:
                return self._error_turn(
                    state,
                    [
                        {
                            "tool": "lookup_order",
                            "arguments": {
                                "order_id": state.order_id,
                                "verification_value": verification,
                            },
                        }
                    ],
                    [],
                    exc,
                )
            tool_calls.append(
                {
                    "tool": "lookup_order",
                    "arguments": {"order_id": state.order_id, "verification_value": verification},
                    "result": _safe_tool_result(lookup),
                }
            )
            if lookup.get("verified") and lookup.get("found"):
                state.verified = True
                state.order = dict(lookup)
                state.actions.append("identity verified")
            elif state.verification_attempts >= 2:
                state.actions.append("identity verification failed twice")
                return self._finish(
                    state,
                    ChatResult(
                        answer="I couldn’t verify the order. I can provide general policy help, or you can contact support from the registered phone number.",
                        decision="refuse",
                        raw="",
                    ),
                    [],
                    tool_calls,
                    evidence_applicable=False,
                )
            else:
                state.actions.append("identity verification failed")
                return self._finish(
                    state,
                    ChatResult(
                        answer="That verification did not match. Please try once more with the phone last four digits or delivery pincode.",
                        decision="clarify",
                        raw="",
                    ),
                    [],
                    tool_calls,
                    evidence_applicable=False,
                )

        prompt = _contextual_question(customer_message, state)
        try:
            generated, evidence = rag.answer(prompt)
        except Exception as exc:
            return self._error_turn(state, [], [], exc)
        if generated.decision == "escalate":
            return self._handover(state, generated, evidence, tool_calls)
        return self._finish(state, generated, evidence, tool_calls)

    def state(self, conversation_id: str) -> ConversationState:
        try:
            return self._conversations[conversation_id][0]
        except KeyError as exc:
            raise KeyError(f"unknown conversation: {conversation_id}") from exc

    def reset(self, conversation_id: str | None = None) -> None:
        if conversation_id is None:
            self._conversations.clear()
        else:
            self._conversations.pop(conversation_id, None)

    def _handover(
        self,
        state: ConversationState,
        generated: ChatResult,
        evidence: list[Any],
        tool_calls: list[dict[str, object]],
    ) -> ConversationTurnResult:
        summary = _handover_summary(state, generated.answer)
        if not state.ticket_attempted:
            state.ticket_attempted = True
            try:
                ticket = self._create_ticket(
                    summary,
                    order_id=state.order_id,
                    reason=generated.answer,
                    intent="human_handover",
                )
            except Exception as exc:
                ticket = {"created": False, "error": f"{type(exc).__name__}: tool failed"}
            tool_calls.append(
                {
                    "tool": "create_ticket",
                    "arguments": {"order_id": state.order_id, "intent": "human_handover"},
                    "result": _safe_tool_result(ticket),
                }
            )
            if ticket.get("created") is True:
                state.ticket_id = str(ticket.get("ticket_id", "")) or None
                state.actions.append("created human-handover ticket")
                answer = "I’m handing this over to a human colleague. Your ticket has been created."
            else:
                state.actions.append("ticket creation failed")
                answer = "I need to hand this over to a human colleague, but ticket creation failed. Please contact support directly."
            generated = ChatResult(answer=answer, decision="escalate", raw=generated.raw)
        elif state.ticket_id:
            generated = ChatResult(
                answer="This has already been handed over to a human colleague; I won’t create a duplicate ticket.",
                decision="escalate",
                raw=generated.raw,
            )
        return self._finish(state, generated, evidence, tool_calls, summary)

    def _finish(
        self,
        state: ConversationState,
        result: ChatResult,
        evidence: list[Any],
        tool_calls: list[dict[str, object]],
        handover_summary: str = "",
        evidence_applicable: bool = True,
    ) -> ConversationTurnResult:
        state.turns.append({"role": "assistant", "content": result.answer})
        return ConversationTurnResult(
            conversation_id=state.conversation_id,
            result=result,
            evidence=evidence_as_dicts(evidence),
            tool_calls=tool_calls,
            evidence_applicable=evidence_applicable,
            handover_summary=handover_summary,
        )

    def _error_turn(
        self,
        state: ConversationState,
        tool_calls: list[dict[str, object]],
        evidence: list[Any],
        exc: Exception,
    ) -> ConversationTurnResult:
        category = str(getattr(exc, "category", "application"))
        message = str(getattr(exc, "safe_message", "conversation processing failed"))
        state.actions.append(f"processing failed: {category}")
        result = ChatResult(
            answer="I’m unable to complete that request right now. Please try again later or contact support.",
            decision="clarify",
            raw="",
        )
        state.turns.append({"role": "assistant", "content": result.answer})
        return ConversationTurnResult(
            conversation_id=state.conversation_id,
            result=result,
            evidence=evidence_as_dicts(evidence),
            tool_calls=tool_calls,
            scored=False,
            error_category=category,
            error_message=message,
        )


def evaluate_conversations(
    rows: Iterable[dict[str, str]],
    agent: ConversationalRAGAgent,
) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: (row["conversation_id"], int(row["turn"])))
    records: list[dict[str, object]] = []
    for row in ordered:
        turn = agent.process_turn(row["conversation_id"], row["customer_message"])
        records.append(
            {
                "conversation_id": row["conversation_id"],
                "turn": int(row["turn"]),
                "scenario": row.get("scenario", ""),
                "result": {
                    "answer": turn.result.answer,
                    "decision": turn.result.decision,
                },
                "tool_calls": turn.tool_calls,
                "retrieved_evidence": turn.evidence,
                "expected_decision": row.get("expected_decision", ""),
                "scored": turn.scored,
                "error_category": turn.error_category,
                "error_message": turn.error_message,
            }
        )
    return records


def _extract_order_id(message: str) -> str | None:
    match = re.search(r"(?<!\d)(\d(?:[\s-]?\d){7})(?!\d)", message)
    return re.sub(r"\D", "", match.group(1)) if match else None


def _extract_verification_value(message: str, order_id: str) -> str | None:
    compact = re.sub(r"\D", "", message)
    compact_order = re.sub(r"\D", "", order_id)
    candidates = re.findall(r"(?<!\d)(\d[\d\s-]{2,7}\d)(?!\d)", message)
    for candidate in candidates:
        value = re.sub(r"\D", "", candidate)
        if value != compact_order and len(value) in {4, 6}:
            return value
    if compact_order in compact and len(compact) > len(compact_order):
        remainder = compact.replace(compact_order, "", 1)
        if len(remainder) in {4, 6}:
            return remainder
    return None


def _needs_order_context(message: str) -> bool:
    lowered = message.casefold()
    return any(
        term in lowered
        for term in (
            "order",
            "ऑर्डर",
            "refund",
            "रिफ़ंड",
            "cancel",
            "कैंसल",
            "delivery",
            "late",
            "item",
        )
    )


def _contextual_question(message: str, state: ConversationState) -> str:
    lowered = message.casefold()
    follow_up = any(term in lowered for term in ("how long", "kitne din", "कितने दिन", "when", "kab", "it", "this"))
    if state.order and (follow_up or _needs_order_context(message)):
        safe_order = {
            key: value
            for key, value in state.order.items()
            if key not in {"phone", "email", "address", "customer_id", "items"}
        }
        return f"Customer message:\n{message}\n\nVerified order context:\n{safe_order}"
    return message


def _handover_summary(state: ConversationState, reason: str) -> str:
    return (
        f"Issue: {state.issue}; intent: human handover; order ID: {state.order_id or 'none'}; "
        f"actions already taken: {', '.join(state.actions) or 'none'}; "
        f"reason: {reason}"
    )


def _merge_issue(existing: str, message: str) -> str:
    return message if not existing else f"{existing} | {message}"


def _safe_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"phone", "email", "address", "customer_id", "items"}
    }


def _load_lookup_order() -> OrderLookup:
    from Capstone_Project_Dataset_FTTT_2026.Capstone_Project_Dataset_FTTT_2026.tools.tools import lookup_order

    return lookup_order


def _load_create_ticket() -> TicketCreator:
    from Capstone_Project_Dataset_FTTT_2026.Capstone_Project_Dataset_FTTT_2026.tools.tools import create_ticket

    return create_ticket
