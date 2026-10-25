import json
import unittest
from pathlib import Path

from voiceops.dataset import DatasetPaths, DatasetRepository, KnowledgeSection
from voiceops.evaluation import evaluate_rag_questions
from voiceops.chatbot import GeminiAPIError
from voiceops.rag import KnowledgeBaseIndex
from voiceops.rag_chatbot import RAGChatbot


class FakeResponse:
    def __init__(self, content):
        self.text = content


class FakeClient:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(json.dumps({"decision": "answer", "answer": "12 months via My Orders"}))


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.repository = DatasetRepository(DatasetPaths.discover(root))

    def test_index_preserves_metadata_and_excludes_superseded(self):
        index = KnowledgeBaseIndex(self.repository.sections())
        self.assertTrue(index.sections)
        self.assertFalse(
            any(section.metadata.get("status") == "superseded" for section in index.sections)
        )
        result = index.retrieve("What is the return window?")
        self.assertTrue(result)
        self.assertTrue(all(item.metadata["status"] == "current" for item in result))

    def test_current_policy_wins_over_superseded_policy(self):
        sections = [
            KnowledgeSection(
                Path("old.md"),
                {
                    "doc_type": "policy",
                    "status": "superseded",
                    "version": "1.0",
                    "superseded_by": "POL-NEW",
                },
                "POL-OLD",
                "Refund window",
                "The old refund window is 30 days.",
            ),
            KnowledgeSection(
                Path("new.md"),
                {
                    "doc_type": "policy",
                    "status": "current",
                    "version": "2.0",
                    "supersedes": "POL-OLD",
                },
                "POL-NEW",
                "Refund window",
                "The current refund window is 7 business days.",
            ),
        ]
        result = KnowledgeBaseIndex(sections).retrieve("What is the refund window?")
        self.assertEqual([item.section_id for item in result], ["POL-NEW"])
        self.assertEqual(result[0].metadata["status"], "current")

    def test_current_policy_gets_precedence_over_lower_authority(self):
        sections = [
            KnowledgeSection(
                Path("faq.md"),
                {"doc_type": "faq", "status": "current"},
                "FAQ-1",
                "Return window",
                "return window policy",
            ),
            KnowledgeSection(
                Path("policy.md"),
                {"doc_type": "policy", "status": "current"},
                "POL-1",
                "Return window",
                "return window policy",
            ),
        ]
        result = KnowledgeBaseIndex(sections).retrieve("return window")
        self.assertEqual(result[0].section_id, "POL-1")

    def test_rag_prompt_contains_provenance_and_evidence(self):
        client = FakeClient()
        chatbot = RAGChatbot(
            client,
            "test-model",
            KnowledgeBaseIndex(self.repository.sections()),
        )
        result, evidence = chatbot.answer("How long is the warranty for an Assured product?")
        self.assertTrue(evidence)
        self.assertIn(evidence[0].section_id, client.calls[0]["messages"][-1]["content"])
        self.assertEqual(result.decision, "answer")
        self.assertIn(str(evidence[0].document), client.calls[0]["messages"][-1]["content"])

    def test_missing_evidence_is_explicitly_passed_to_generator(self):
        client = FakeClient()
        chatbot = RAGChatbot(
            client,
            "test-model",
            KnowledgeBaseIndex(self.repository.sections()),
        )
        result, evidence = chatbot.answer("What is the moon zorbaflint?")
        self.assertFalse(evidence)
        prompt = client.calls[0]["messages"][-1]["content"]
        self.assertIn("No matching knowledge-base section was retrieved.", prompt)
        self.assertEqual(result.decision, "answer")

    def test_conversational_context_does_not_displace_current_policy_query(self):
        chatbot = RAGChatbot(
            FakeClient(),
            "test-model",
            KnowledgeBaseIndex(self.repository.sections()),
        )
        _, evidence = chatbot.answer(
            "Customer message:\nAur refund kitne din mein aayega?\n\n"
            "Verified order context:\n"
            "{'order_id': '18599253', 'status': 'delivered', "
            "'customer_first_name': 'REDACTED'}"
        )
        ids = {item.section_id for item in evidence}
        self.assertTrue({"REF-2", "RET-7"}.issubset(ids))

    def test_low_information_opening_retrieves_opening_guidance(self):
        evidence = KnowledgeBaseIndex(self.repository.sections()).retrieve(
            "Hello, mujhe order ke baare mein help chahiye."
        )
        self.assertIn("STY-1", {item.section_id for item in evidence})

    def test_rag_evaluation_records_evidence(self):
        chatbot = RAGChatbot(
            FakeClient(),
            "test-model",
            KnowledgeBaseIndex(self.repository.sections()),
        )
        records = evaluate_rag_questions(
            [
                {
                    "id": "Q1",
                    "question": "Warranty?",
                    "expected_decision": "answer",
                    "must_include": "12 ; My Orders / app",
                    "gold_evidence": "WAR-2",
                }
            ],
            chatbot,
        )
        self.assertTrue(records[0].retrieved_evidence)
        self.assertIsNotNone(records[0].evidence_recall)

    def test_answerable_question_is_scored_from_retrieved_evidence(self):
        chatbot = RAGChatbot(
            FakeClient(),
            "test-model",
            KnowledgeBaseIndex(self.repository.sections()),
        )
        question = {
            "id": "Q-WARRANTY",
            "question": "How long is the Assured warranty?",
            "expected_decision": "answer",
            "acceptable_decisions": "answer",
            "answerable": "True",
            "must_include": "12 ; My Orders / app",
            "must_not_include": "no warranty",
            "gold_evidence": "WAR-2",
        }
        record = evaluate_rag_questions([question], chatbot)[0]
        self.assertTrue(record.scored)
        self.assertTrue(record.correct)
        self.assertTrue(record.evidence_recall)
        self.assertTrue(
            any(item["section_id"] == "WAR-2" for item in record.retrieved_evidence)
        )

    def test_historical_superseded_evidence_does_not_reduce_active_recall(self):
        question = next(
            item for item in self.repository.practice_questions() if item["id"] == "PQ0019"
        )
        record = evaluate_rag_questions(
            [question],
            RAGChatbot(
                FakeClient(),
                "test-model",
                KnowledgeBaseIndex(self.repository.sections()),
            ),
        )[0]
        self.assertTrue(record.evidence_recall)

    def test_unsupported_question_accepts_safe_escalation(self):
        class SafeEscalationClient:
            def create(self, **kwargs):
                return FakeResponse(
                    json.dumps(
                        {
                            "decision": "escalate",
                            "answer": (
                                "I do not have information about that. "
                                "I can connect you to a colleague."
                            ),
                        }
                    )
                )

        question = next(
            item
            for item in self.repository.practice_questions()
            if item["answerable"] == "False" and "escalate" in item["acceptable_decisions"]
        )
        chatbot = RAGChatbot(
            SafeEscalationClient(),
            "test-model",
            KnowledgeBaseIndex(self.repository.sections()),
        )
        record = evaluate_rag_questions([question], chatbot)[0]
        self.assertTrue(record.scored)
        self.assertTrue(record.decision_correct)
        self.assertTrue(record.forbidden_claims_absent)
        self.assertTrue(record.required_facts_found)
        self.assertTrue(record.correct)

    def test_identical_mock_response_is_scored_deterministically(self):
        question = {
            "id": "Q-DETERMINISTIC",
            "question": "How long is the Assured warranty?",
            "expected_decision": "answer",
            "acceptable_decisions": "answer",
            "must_include": "12 ; My Orders / app",
            "gold_evidence": "WAR-2",
        }
        first = evaluate_rag_questions(
            [question],
            RAGChatbot(
                FakeClient(),
                "test-model",
                KnowledgeBaseIndex(self.repository.sections()),
            ),
        )[0]
        second = evaluate_rag_questions(
            [question],
            RAGChatbot(
                FakeClient(),
                "test-model",
                KnowledgeBaseIndex(self.repository.sections()),
            ),
        )[0]
        self.assertEqual(first.correct, second.correct)
        self.assertEqual(first.required_facts_found, second.required_facts_found)
        self.assertEqual(first.evidence_recall, second.evidence_recall)

    def test_rag_evaluation_preserves_structured_api_errors(self):
        class QuotaClient:
            def create(self, **kwargs):
                raise GeminiAPIError(
                    "quota-exhausted",
                    "Gemini quota is exhausted",
                    retryable=False,
                    status_code=429,
                )

        chatbot = RAGChatbot(
            QuotaClient(),
            "test-model",
            KnowledgeBaseIndex(self.repository.sections()),
        )
        records = evaluate_rag_questions(
            [{"id": "Q1", "question": "Warranty?", "expected_decision": "answer"}],
            chatbot,
        )
        self.assertFalse(records[0].scored)
        self.assertEqual(records[0].error_category, "quota-exhausted")
        self.assertEqual(records[0].error_status_code, 429)

    def test_pq0002_retrieves_warranty_policy(self):
        question = next(
            item for item in self.repository.practice_questions() if item["id"] == "PQ0002"
        )
        evidence = KnowledgeBaseIndex(self.repository.sections()).retrieve(question["question"])
        self.assertIn("WAR-2", {item.section_id for item in evidence})

    def test_late_express_delivery_retrieves_compensation_policy(self):
        question = next(
            item for item in self.repository.practice_questions() if item["id"] == "PQ0047"
        )
        evidence = KnowledgeBaseIndex(self.repository.sections()).retrieve(question["question"])
        self.assertIn("DEL-4", {item.section_id for item in evidence})

    def test_cod_return_retrieves_timeline_and_cod_policies(self):
        question = next(
            item for item in self.repository.practice_questions() if item["id"] == "PQ0050"
        )
        evidence = KnowledgeBaseIndex(self.repository.sections()).retrieve(question["question"])
        ids = {item.section_id for item in evidence}
        self.assertTrue({"REF-2", "REF-3"}.issubset(ids))

    def test_privacy_request_retrieves_safety_guidance(self):
        question = next(
            item for item in self.repository.practice_questions() if item["id"] == "PQ0060"
        )
        evidence = KnowledgeBaseIndex(self.repository.sections()).retrieve(question["question"])
        ids = {item.section_id for item in evidence}
        self.assertTrue({"ESC-4", "IDV-4"}.issubset(ids))

    def test_hindi_return_window_retrieves_current_policy(self):
        question = next(
            item for item in self.repository.practice_questions() if item["id"] == "PQ0018"
        )
        evidence = KnowledgeBaseIndex(self.repository.sections()).retrieve(question["question"])
        self.assertIn("RET-2", {item.section_id for item in evidence})

    def test_returnable_category_paraphrase_retrieves_nonreturnable_policy(self):
        question = next(
            item for item in self.repository.practice_questions() if item["id"] == "PQ0039"
        )
        evidence = KnowledgeBaseIndex(self.repository.sections()).retrieve(question["question"])
        self.assertIn("RET-5", {item.section_id for item in evidence})

    def test_all_practice_questions_have_active_evidence_coverage(self):
        index = KnowledgeBaseIndex(self.repository.sections())
        covered = 0
        for question in self.repository.practice_questions():
            gold = {
                item.strip()
                for item in question["gold_evidence"].split(" ; ")
                if item.strip()
                and not item.startswith("products.csv:")
                and item != "RETv3-2"
            }
            retrieved = {
                item.section_id for item in index.retrieve(question["question"])
            }
            covered += gold.issubset(retrieved)
        self.assertEqual(covered, 52)


if __name__ == "__main__":
    unittest.main()
