"""Deterministic scoring and reporting for multi-turn conversations."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .conversation import ConversationTurnResult, ConversationalRAGAgent


@dataclass(frozen=True)
class ConversationScore:
    conversation_id: str
    turn: int
    scenario: str
    scored: bool
    decision_correct: bool | None
    required_facts_found: bool | None
    forbidden_claims_absent: bool | None
    tool_calls_correct: bool | None
    identity_behavior_correct: bool | None
    ticket_behavior_correct: bool | None
    handover_behavior_correct: bool | None
    evidence_recall: bool | None
    context_behavior_correct: bool | None
    evidence_applicable: bool | None
    error_category: str
    diagnostics: list[str] = field(default_factory=list)
    decision: str = ""
    expected_decision: str = ""
    actual_tools: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    retrieved_section_ids: list[str] = field(default_factory=list)


def score_conversations(
    rows: Iterable[dict[str, str]],
    agent: ConversationalRAGAgent,
) -> list[ConversationScore]:
    ordered = sorted(rows, key=lambda row: (row["conversation_id"], int(row["turn"])))
    records: list[ConversationScore] = []
    seen_order_ids: dict[str, set[str]] = defaultdict(set)
    for row in ordered:
        turn = agent.process_turn(row["conversation_id"], row["customer_message"])
        score = _score_turn(
            row,
            turn,
            _active_section_ids(agent),
            seen_order_ids,
            agent,
        )
        records.append(score)
        for call in turn.tool_calls:
            order_id = str(call.get("arguments", {}).get("order_id", ""))
            if order_id:
                seen_order_ids[row["conversation_id"]].add(order_id)
    return records


def build_report(
    rows: Iterable[dict[str, str]],
    scores: Iterable[ConversationScore],
) -> dict[str, Any]:
    rows_by_key = {(row["conversation_id"], int(row["turn"])): row for row in rows}
    score_list = list(scores)
    return {
        "schema_version": 1,
        "scope": {
            "turns": len(score_list),
            "conversations": len({score.conversation_id for score in score_list}),
            "evaluation_mode": "deterministic_mock",
            "live_model_accuracy_measured": False,
        },
        "aggregate": _aggregate(score_list),
        "scenarios": _scenario_breakdown(score_list),
        "failed_or_unscored": [
            _safe_score(score, rows_by_key.get((score.conversation_id, score.turn), {}))
            for score in score_list
            if not score.scored or any(
                value is False
                for value in (
                    score.decision_correct,
                    score.required_facts_found,
                    score.forbidden_claims_absent,
                    score.tool_calls_correct,
                    score.evidence_recall,
                )
                if value is not None
            )
        ],
        "priority_failure_categories": _priority_failures(score_list),
    }


def write_report(
    output_dir: Path,
    rows: Iterable[dict[str, str]],
    scores: Iterable[ConversationScore],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_list = list(scores)
    report = build_report(rows, score_list)
    per_turn_path = output_dir / "step3_conversation_per_turn.json"
    aggregate_path = output_dir / "step3_conversation_report.json"
    per_turn_path.write_text(
        json.dumps([asdict(score) for score in score_list], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    aggregate_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return per_turn_path, aggregate_path


def _score_turn(
    row: dict[str, str],
    turn: ConversationTurnResult,
    active_ids: set[str],
    seen_order_ids: dict[str, set[str]],
    agent: ConversationalRAGAgent,
) -> ConversationScore:
    diagnostics: list[str] = []
    if not turn.scored:
        diagnostics.append("processing failure was left unscored")
        return ConversationScore(
            conversation_id=row["conversation_id"],
            turn=int(row["turn"]),
            scenario=row.get("scenario", ""),
            scored=False,
            decision_correct=None,
            required_facts_found=None,
            forbidden_claims_absent=None,
            tool_calls_correct=None,
            identity_behavior_correct=None,
            ticket_behavior_correct=None,
            handover_behavior_correct=None,
            evidence_recall=None,
            context_behavior_correct=None,
            evidence_applicable=None,
            error_category=turn.error_category or "application",
            diagnostics=diagnostics,
            decision=turn.result.decision,
            expected_decision=row.get("expected_decision", ""),
        )

    answer = turn.result.answer
    expected_decision = row.get("expected_decision", "").strip()
    allowed = _allowed_decisions(row)
    decision_correct = bool(expected_decision) and turn.result.decision in allowed
    required = _facts_present(row.get("must_include", ""), answer)
    forbidden = _forbidden_absent(row.get("must_not_include", ""), answer)
    expected_tools = _split_tools(row.get("expected_tool_calls", ""))
    actual_tools = [_canonical_tool_call(call) for call in turn.tool_calls]
    tools_correct = actual_tools == expected_tools
    evidence_ids = sorted(
        str(item.get("section_id"))
        for item in turn.evidence
        if item.get("section_id")
    )
    gold_ids = {
        item.strip()
        for item in row.get("gold_evidence", "").split(" ; ")
        if item.strip() and not item.startswith("products.csv:")
    }
    active_gold = gold_ids & active_ids
    evidence_recall = (
        active_gold.issubset(set(evidence_ids))
        if active_gold and turn.evidence_applicable
        else None
    )
    evidence_applicable = turn.evidence_applicable if active_gold else None
    identity = _identity_behavior(row, turn)
    ticket = _ticket_behavior(row, turn)
    handover = _handover_behavior(row, turn)
    context = _context_behavior(row, turn, seen_order_ids, agent)
    for label, value in (
        ("decision", decision_correct),
        ("required facts", required),
        ("forbidden claims", forbidden),
        ("tool calls", tools_correct),
        ("evidence recall", evidence_recall),
        ("identity behavior", identity),
        ("ticket behavior", ticket),
        ("handover behavior", handover),
        ("context behavior", context),
    ):
        if value is False:
            diagnostics.append(f"{label} mismatch")
    return ConversationScore(
        conversation_id=row["conversation_id"],
        turn=int(row["turn"]),
        scenario=row.get("scenario", ""),
        scored=True,
        decision_correct=decision_correct,
        required_facts_found=required,
        forbidden_claims_absent=forbidden,
        tool_calls_correct=tools_correct,
        identity_behavior_correct=identity,
        ticket_behavior_correct=ticket,
        handover_behavior_correct=handover,
        evidence_recall=evidence_recall,
        context_behavior_correct=context,
        evidence_applicable=evidence_applicable,
        error_category="",
        diagnostics=diagnostics,
        decision=turn.result.decision,
        expected_decision=expected_decision,
        actual_tools=actual_tools,
        expected_tools=expected_tools,
        retrieved_section_ids=evidence_ids,
    )


def _aggregate(scores: list[ConversationScore]) -> dict[str, dict[str, int]]:
    names = (
        "decision_correct",
        "required_facts_found",
        "forbidden_claims_absent",
        "tool_calls_correct",
        "identity_behavior_correct",
        "ticket_behavior_correct",
        "handover_behavior_correct",
        "evidence_recall",
        "context_behavior_correct",
    )
    result: dict[str, dict[str, int]] = {
        "scored_turns": {"numerator": sum(score.scored for score in scores), "denominator": len(scores)}
    }
    for name in names:
        values = [getattr(score, name) for score in scores if getattr(score, name) is not None]
        result[name] = {"numerator": sum(values), "denominator": len(values)}
    result["unscored_errors"] = {
        "numerator": sum(not score.scored for score in scores),
        "denominator": len(scores),
    }
    return result


def _scenario_breakdown(scores: list[ConversationScore]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ConversationScore]] = defaultdict(list)
    for score in scores:
        grouped[score.scenario].append(score)
    return {scenario: _aggregate(items) for scenario, items in sorted(grouped.items())}


def _priority_failures(scores: list[ConversationScore]) -> list[dict[str, Any]]:
    counts = Counter(
        diagnostic
        for score in scores
        for diagnostic in score.diagnostics
    )
    return [
        {"category": category, "count": count}
        for category, count in counts.most_common()
    ]


def _safe_score(score: ConversationScore, row: dict[str, str]) -> dict[str, Any]:
    data = asdict(score)
    data["customer_message"] = _safe_text(row.get("customer_message", ""))
    return data


def _allowed_decisions(row: dict[str, str]) -> set[str]:
    raw = row.get("acceptable_decisions", "").strip() or row.get("expected_decision", "")
    return {item.strip() for item in raw.split(" ; ") if item.strip()}


def _facts_present(requirements: str, answer: str) -> bool:
    normalized_answer = _normalise(answer)
    for requirement in requirements.split(" ; "):
        alternatives = [part.strip() for part in requirement.split(" / ") if part.strip()]
        if alternatives and not any(
            _fact_matches(part, normalized_answer) for part in alternatives
        ):
            return False
    return True


def _forbidden_absent(requirements: str, answer: str) -> bool:
    normalized_answer = _normalise(answer)
    return not any(
        _normalise(part) in normalized_answer
        for part in requirements.split(" ; ")
        if part.strip()
    )


def _split_tools(value: str) -> list[str]:
    return [_canonical_expected_tool(item) for item in value.split(" ; ") if item.strip()]


def _canonical_tool_call(call: dict[str, object]) -> str:
    name = str(call.get("tool", ""))
    arguments = call.get("arguments", {})
    if name == "lookup_order" and isinstance(arguments, dict):
        values = {
            key: _normalise_identifier(str(value))
            for key, value in arguments.items()
            if value is not None
        }
        return _format_tool(name, values)
    if name == "create_ticket":
        return "create_ticket()"
    return name


def _canonical_expected_tool(value: str) -> str:
    match = re.fullmatch(r"\s*([a-z_]+)\s*(?:\((.*)\))?\s*", value)
    if not match:
        return value.strip()
    name, arguments_text = match.groups()
    if not arguments_text:
        return f"{name}()"
    arguments = _parse_arguments(arguments_text)
    return _format_tool(name, arguments)


def _parse_arguments(value: str) -> dict[str, str]:
    arguments: dict[str, str] = {}
    for part in value.split(","):
        if "=" not in part:
            continue
        key, argument = part.split("=", 1)
        arguments[key.strip()] = _normalise_identifier(argument)
    return arguments


def _format_tool(name: str, arguments: dict[str, str]) -> str:
    if not arguments:
        return f"{name}()"
    return f"{name}(" + ", ".join(
        f"{key}={arguments[key]}" for key in sorted(arguments)
    ) + ")"


def _identity_behavior(row: dict[str, str], turn: ConversationTurnResult) -> bool | None:
    calls = [call for call in turn.tool_calls if call.get("tool") == "lookup_order"]
    if not calls:
        return None if "lookup_order" not in row.get("expected_tool_calls", "") else False
    result = calls[-1].get("result", {})
    if not isinstance(result, dict):
        return False
    expected_verification = row.get("expected_tool_calls", "")
    expects_value = "verification_value=" in expected_verification
    return bool(result.get("verified") and result.get("found")) == expects_value


def _ticket_behavior(row: dict[str, str], turn: ConversationTurnResult) -> bool | None:
    expected = "create_ticket()" in row.get("expected_tool_calls", "")
    calls = [call for call in turn.tool_calls if call.get("tool") == "create_ticket"]
    if not expected:
        return False if calls else None
    if not calls:
        return False
    result = calls[-1].get("result", {})
    return isinstance(result, dict) and result.get("created") is True


def _handover_behavior(row: dict[str, str], turn: ConversationTurnResult) -> bool | None:
    expected = row.get("expected_decision", "") == "escalate"
    if not expected:
        return None
    return (
        turn.result.decision == "escalate"
        and "create_ticket()" in row.get("expected_tool_calls", "")
        and bool(turn.handover_summary)
    )


def _context_behavior(
    row: dict[str, str],
    turn: ConversationTurnResult,
    seen_order_ids: dict[str, set[str]],
    agent: ConversationalRAGAgent,
) -> bool | None:
    expected_order = row.get("order_id", "").strip()
    prior_turn = int(row.get("turn", "1")) > 1
    if not expected_order or not prior_turn:
        return None
    observed = {
        str(call.get("arguments", {}).get("order_id", ""))
        for call in turn.tool_calls
        if isinstance(call.get("arguments"), dict) and call.get("arguments", {}).get("order_id")
    }
    try:
        state_order = agent.state(row["conversation_id"]).order_id
    except KeyError:
        return False
    return state_order == expected_order and (
        not observed or expected_order in observed
    )


def _active_section_ids(agent: ConversationalRAGAgent) -> set[str]:
    sections: set[str] = set()
    for _, rag in agent._conversations.values():
        sections.update(item.section_id for item in rag.index.sections)
    return sections


def _normalise(value: str) -> str:
    value = "".join(
        str(unicodedata.digit(char)) if char.isdigit() else char
        for char in value
    )
    value = value.casefold().replace("-", " ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _normalise_identifier(value: str) -> str:
    return re.sub(r"\s+", "", _normalise(value))


def _fact_matches(fact: str, normalized_answer: str) -> bool:
    normalized_fact = _normalise(fact)
    if normalized_fact in normalized_answer:
        return True
    date_match = re.fullmatch(r"(\d{4})\s+(\d{1,2})\s+(\d{1,2})", normalized_fact)
    if date_match:
        year, month, day = date_match.groups()
        month_name = {
            "01": "january", "02": "february", "03": "march",
            "04": "april", "05": "may", "06": "june",
            "07": "july", "08": "august", "09": "september",
            "10": "october", "11": "november", "12": "december",
        }[month.zfill(2)]
        return f"{day} {month_name} {year}" in normalized_answer
    return False


def _safe_text(value: str) -> str:
    return re.sub(r"(?<!\d)(?:\d[\s-]?){4,}(?!\d)", "[REDACTED_NUMBER]", value)
