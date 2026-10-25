import json
import unittest
from pathlib import Path

from voiceops.chatbot import ChatResult
from voiceops.conversation import ConversationalRAGAgent, evaluate_conversations
from voiceops.dataset import DatasetPaths, DatasetRepository
from voiceops.rag import KnowledgeBaseIndex
from voiceops.rag_chatbot import RAGChatbot


class FakeRAG:
    def __init__(self, decision="answer", answer="I used the supplied policy."):
        self.decision = decision
        self.answer_text = answer
        self.calls = []

    def answer(self, question):
        self.calls.append(question)
        return (
            ChatResult(
                answer=self.answer_text,
                decision=self.decision,
                raw=json.dumps({"decision": self.decision, "answer": self.answer_text}),
            ),
            [],
        )


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.lookups = []
        self.tickets = []

        def lookup(order_id, verification_value=None):
            self.lookups.append((order_id, verification_value))
            if verification_value is None:
                return {"verification_required": True, "order_id": order_id}
            if verification_value == "1234":
                return {
                    "found": True,
                    "verified": True,
                    "order_id": order_id,
                    "status": "delivered",
                    "total_inr": 1500,
                    "contacts_last_7_days": 0,
                    "phone": "private",
                }
            return {"verified": False, "verification_required": True}

        def ticket(summary, order_id=None, reason=None, intent=None):
            self.tickets.append((summary, order_id, reason, intent))
            return {"created": True, "ticket_id": "TKT-1"}

        self.lookup = lookup
        self.ticket = ticket

    def agent(self, decision="answer", answer="I used the supplied policy."):
        self.rags = []

        def factory():
            rag = FakeRAG(decision, answer)
            self.rags.append(rag)
            return rag

        return ConversationalRAGAgent(
            factory,
            lookup_order=self.lookup,
            create_ticket=self.ticket,
        )

    def test_context_is_isolated_and_order_is_retained(self):
        agent = self.agent()
        first = agent.process_turn("A", "I need help with order 12345678")
        second = agent.process_turn("A", "The last four digits are 1234")
        other = agent.process_turn("B", "I need help with an order")
        self.assertEqual(first.result.decision, "clarify")
        self.assertTrue(second.result.answer)
        self.assertEqual(other.result.decision, "clarify")
        self.assertEqual(agent.state("A").order_id, "12345678")
        self.assertIsNone(agent.state("B").order_id)
        self.assertEqual(len(self.rags), 2)

    def test_failed_verification_allows_one_retry_then_refuses(self):
        agent = self.agent()
        agent.process_turn("A", "Order 12345678")
        failed = agent.process_turn("A", "Last four are 9999")
        refused = agent.process_turn("A", "Pincode 999999")
        self.assertEqual(failed.result.decision, "clarify")
        self.assertEqual(refused.result.decision, "refuse")
        self.assertNotIn("1500", refused.result.answer)
        self.assertEqual(self.lookups, [("12345678", None), ("12345678", "9999"), ("12345678", "999999")])

    def test_follow_up_receives_verified_context_without_private_fields(self):
        agent = self.agent()
        agent.process_turn("A", "Order 12345678")
        agent.process_turn("A", "Phone last four 1234")
        follow_up = agent.process_turn("A", "How long will that take?")
        self.assertIn("Verified order context", self.rags[0].calls[-1])
        self.assertIn("status", self.rags[0].calls[-1])
        self.assertNotIn("private", self.rags[0].calls[-1])
        self.assertEqual(follow_up.result.decision, "answer")

    def test_escalation_creates_ticket_once_and_returns_handover_summary(self):
        agent = self.agent(decision="escalate", answer="A human colleague should review this.")
        first = agent.process_turn("A", "Please connect me to a human")
        second = agent.process_turn("A", "Please connect me to a human again")
        self.assertEqual(first.result.decision, "escalate")
        self.assertEqual(len(self.tickets), 1)
        self.assertIn("Issue:", first.handover_summary)
        self.assertIn("human handover", self.tickets[0][0])
        self.assertIn("already been handed over", second.result.answer)

    def test_ticket_failure_does_not_claim_success(self):
        def failed_ticket(*args, **kwargs):
            return {"created": False, "error": "backend unavailable"}

        agent = ConversationalRAGAgent(
            lambda: FakeRAG("escalate", "Please hand this over."),
            lookup_order=self.lookup,
            create_ticket=failed_ticket,
        )
        result = agent.process_turn("A", "I need a human")
        self.assertEqual(result.result.decision, "escalate")
        self.assertIn("failed", result.result.answer)
        self.assertNotIn("created", result.result.answer.casefold())

    def test_lookup_failure_is_unscored_and_does_not_expose_order_details(self):
        def failed_lookup(*args, **kwargs):
            raise RuntimeError("lookup backend unavailable")

        agent = ConversationalRAGAgent(
            lambda: FakeRAG(),
            lookup_order=failed_lookup,
            create_ticket=self.ticket,
        )
        result = agent.process_turn("A", "Order 12345678")
        self.assertFalse(result.scored)
        self.assertEqual(result.error_category, "application")
        self.assertNotIn("12345678", result.result.answer)

    def test_llm_failure_is_unscored_and_preserves_failure_category(self):
        class FailedRAG:
            def answer(self, question):
                error = RuntimeError("temporarily unavailable")
                error.category = "service"
                error.safe_message = "Gemini service unavailable"
                raise error

        agent = ConversationalRAGAgent(
            lambda: FailedRAG(),
            lookup_order=self.lookup,
            create_ticket=self.ticket,
        )
        result = agent.process_turn("A", "What is the return policy?")
        self.assertFalse(result.scored)
        self.assertEqual(result.error_category, "service")
        self.assertIn("unable", result.result.answer)

    def test_evaluation_processes_turns_in_order_and_preserves_evidence_shape(self):
        rows = [
            {
                "conversation_id": "B",
                "turn": "2",
                "scenario": "test",
                "customer_message": "last four 1234",
                "expected_decision": "answer",
            },
            {
                "conversation_id": "B",
                "turn": "1",
                "scenario": "test",
                "customer_message": "Order 12345678",
                "expected_decision": "clarify",
            },
        ]
        records = evaluate_conversations(rows, self.agent())
        self.assertEqual([record["turn"] for record in records], [1, 2])
        self.assertIn("retrieved_evidence", records[0])
        self.assertEqual(records[1]["tool_calls"][0]["tool"], "lookup_order")

    def test_real_kb_rag_factory_preserves_evidence_provenance(self):
        root = Path(r"C:\Users\Admin\OneDrive\Desktop\Projects\Capstone")
        repository = DatasetRepository(DatasetPaths.discover(root))
        index = KnowledgeBaseIndex(repository.sections())

        class Client:
            def create(self, **kwargs):
                return type(
                    "Response",
                    (),
                    {"text": json.dumps({"decision": "answer", "answer": "The policy applies."})},
                )()

        rag = RAGChatbot(Client(), "offline", index)
        result, evidence = rag.answer("Can I use two coupons?")
        self.assertEqual(result.decision, "answer")
        self.assertTrue(evidence)
        self.assertTrue(all(item.section_id and item.document for item in evidence))


if __name__ == "__main__":
    unittest.main()
