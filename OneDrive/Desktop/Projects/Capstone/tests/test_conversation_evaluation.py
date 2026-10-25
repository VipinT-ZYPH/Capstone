import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from voiceops.chatbot import ChatResult
from voiceops.conversation import ConversationalRAGAgent
from voiceops.conversation_evaluation import score_conversations, write_report
from voiceops.rag import RetrievedEvidence


def _load_step3_harness():
    path = Path(__file__).parents[1] / "scripts" / "run_step3_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_step3_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeRAG:
    def __init__(self, result, evidence=None):
        self.result = result
        self.evidence = evidence or []
        self.index = type(
            "Index",
            (),
            {
                "sections": [
                    type(
                        "Section",
                        (),
                        {
                            "section_id": (
                                item.section_id
                                if hasattr(item, "section_id")
                                else item["section_id"]
                            )
                        },
                    )
                    for item in self.evidence
                ]
            },
        )()

    def answer(self, question):
        return self.result, self.evidence


class ConversationEvaluationTests(unittest.TestCase):
    def agent(self, result, evidence=None, lookup=None, ticket=None):
        return ConversationalRAGAgent(
            lambda: FakeRAG(result, evidence),
            lookup_order=lookup or (lambda order_id, verification_value=None: {"verification_required": True}),
            create_ticket=ticket or (lambda *args, **kwargs: {"created": True, "ticket_id": "TKT-1"}),
        )

    def test_scores_decision_facts_forbidden_claims_and_evidence(self):
        result = ChatResult("The deadline is 27 October 2026.", "answer", "{}")
        evidence = [
            RetrievedEvidence(
                section_id="RET-2",
                title="Returns",
                text="return policy",
                document="pol_returns_v4.md",
                metadata={"status": "current"},
                score=1.0,
            )
        ]
        rows = [
            {
                "conversation_id": "C1",
                "turn": "1",
                "scenario": "returns",
                "customer_message": "Can I return it?",
                "expected_decision": "answer",
                "must_include": "27 October 2026",
                "must_not_include": "I approved",
                "expected_tool_calls": "",
                "gold_evidence": "RET-2",
            }
        ]
        agent = self.agent(result, evidence)
        scores = score_conversations(rows, agent)
        self.assertTrue(scores[0].decision_correct)
        self.assertTrue(scores[0].required_facts_found)
        self.assertTrue(scores[0].forbidden_claims_absent)
        self.assertTrue(scores[0].evidence_recall)

    def test_tool_mismatch_and_missing_fields_are_scored_explicitly(self):
        result = ChatResult("Please provide verification.", "clarify", "{}")
        rows = [
            {
                "conversation_id": "C1",
                "turn": "1",
                "scenario": "verification",
                "customer_message": "Order 12345678",
                "expected_decision": "answer",
                "expected_tool_calls": "lookup_order(order_id=12345678, verification_value=1234)",
            }
        ]
        scores = score_conversations(rows, self.agent(result))
        self.assertFalse(scores[0].decision_correct)
        self.assertFalse(scores[0].tool_calls_correct)
        self.assertTrue(scores[0].required_facts_found)
        self.assertTrue(scores[0].forbidden_claims_absent)

    def test_tool_argument_order_and_spacing_are_equivalent(self):
        result = ChatResult("Verified.", "answer", "{}")

        def lookup(order_id, verification_value=None):
            return {"found": True, "verified": True}

        rows = [
            {
                "conversation_id": "C1",
                "turn": "1",
                "scenario": "verification",
                "customer_message": "Order 12345678, phone 1234",
                "expected_decision": "answer",
                "expected_tool_calls": (
                    "lookup_order( verification_value = 1234, order_id = 12345678 )"
                ),
            }
        ]
        scores = score_conversations(
            rows,
            self.agent(
                result,
                lookup=lookup,
            ),
        )
        self.assertTrue(scores[0].tool_calls_correct)

    def test_evidence_metric_is_not_applicable_without_active_gold(self):
        result = ChatResult("General guidance.", "answer", "{}")
        rows = [
            {
                "conversation_id": "C1",
                "turn": "1",
                "scenario": "general",
                "customer_message": "What can you help with?",
                "expected_decision": "answer",
                "gold_evidence": "",
            }
        ]
        scores = score_conversations(rows, self.agent(result))
        self.assertIsNone(scores[0].evidence_recall)

    def test_evidence_metric_excludes_tool_only_identity_turn(self):
        result = ChatResult(
            "Please provide the last four digits.",
            "clarify",
            "{}",
        )

        def lookup(order_id, verification_value=None):
            return {"verification_required": True}

        rows = [
            {
                "conversation_id": "C1",
                "turn": "1",
                "scenario": "verification",
                "customer_message": "Order 12345678",
                "expected_decision": "clarify",
                "expected_tool_calls": "lookup_order(order_id=12345678)",
                "gold_evidence": "IDV-2",
            }
        ]
        score = score_conversations(rows, self.agent(result, lookup=lookup))[0]
        self.assertFalse(score.evidence_applicable)
        self.assertIsNone(score.evidence_recall)

    def test_non_applicable_identity_ticket_and_handover_are_excluded(self):
        result = ChatResult("General guidance.", "answer", "{}")
        rows = [
            {
                "conversation_id": "C1",
                "turn": "1",
                "scenario": "general",
                "customer_message": "What can you help with?",
                "expected_decision": "answer",
            }
        ]
        score = score_conversations(rows, self.agent(result))[0]
        self.assertIsNone(score.identity_behavior_correct)
        self.assertIsNone(score.ticket_behavior_correct)
        self.assertIsNone(score.handover_behavior_correct)

    def test_date_format_equivalence_follows_dataset_semantics(self):
        result = ChatResult("The deadline is 27 October 2026.", "answer", "{}")
        rows = [
            {
                "conversation_id": "C1",
                "turn": "1",
                "scenario": "returns",
                "customer_message": "When is the deadline?",
                "expected_decision": "answer",
                "must_include": "2026-10-27",
            }
        ]
        score = score_conversations(rows, self.agent(result))[0]
        self.assertTrue(score.required_facts_found)

    def test_mock_policy_text_human_reference_does_not_escalate(self):
        harness = _load_step3_harness()
        client = harness.MockClient()
        response = client.create(
            model="offline",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Customer question:\nCan I use two coupons?\n\n"
                        "Retrieved evidence:\n"
                        "[ESC-1] Human handover policy: ask for a human when needed."
                    ),
                }
            ],
        )
        self.assertEqual(json.loads(response.text)["decision"], "answer")

    def test_mock_escalates_only_for_current_turn_request(self):
        harness = _load_step3_harness()
        client = harness.MockClient()
        response = client.create(
            model="offline",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Customer question:\nPlease let me speak to a human agent.\n\n"
                        "Retrieved evidence:\n[FAQ-4] Human support is available."
                    ),
                }
            ],
        )
        self.assertEqual(json.loads(response.text)["decision"], "escalate")

    def test_api_failure_is_unscored_not_incorrect(self):
        class FailedRAG:
            index = type("Index", (), {"sections": []})()

            def answer(self, question):
                error = RuntimeError("service failed")
                error.category = "service"
                error.safe_message = "service unavailable"
                raise error

        agent = ConversationalRAGAgent(
            lambda: FailedRAG(),
            lookup_order=lambda *args, **kwargs: {},
            create_ticket=lambda *args, **kwargs: {"created": False},
        )
        rows = [
            {
                "conversation_id": "C1",
                "turn": "1",
                "scenario": "failure",
                "customer_message": "What is the policy?",
                "expected_decision": "answer",
            }
        ]
        scores = score_conversations(rows, agent)
        self.assertFalse(scores[0].scored)
        self.assertEqual(scores[0].error_category, "service")
        self.assertIsNone(scores[0].decision_correct)

    def test_report_is_reproducible_and_redacts_numbers(self):
        result = ChatResult("Order 12345678 is ready.", "answer", "{}")
        rows = [
            {
                "conversation_id": "C1",
                "turn": "1",
                "scenario": "status",
                "customer_message": "What is the policy?",
                "expected_decision": "answer",
            }
        ]
        scores = score_conversations(rows, self.agent(result))
        with tempfile.TemporaryDirectory() as directory:
            paths = write_report(Path(directory), rows, scores)
            report = json.loads(paths[1].read_text(encoding="utf-8"))
            self.assertEqual(report["aggregate"]["decision_correct"], {"numerator": 1, "denominator": 1})
            self.assertNotIn("12345678", json.dumps(report))
            first = paths[0].read_text(encoding="utf-8")
            second = paths[0].read_text(encoding="utf-8")
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
