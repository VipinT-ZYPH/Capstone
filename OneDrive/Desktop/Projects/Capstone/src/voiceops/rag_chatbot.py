"""RAG-grounded ZentroMart chatbot."""

from __future__ import annotations

import json
from typing import Any

from .chatbot import (
    ChatCompletionClient,
    ChatResult,
    SYSTEM_PROMPT,
    _parse_result,
    _response_text,
)
from .rag import KnowledgeBaseIndex, RetrievedEvidence

RAG_SYSTEM_PROMPT = SYSTEM_PROMPT + """
You are now answering with retrieved ZentroMart knowledge-base evidence.
Use only the evidence enclosed in the user message. Do not use general
knowledge or fill gaps with guesses. Current policy documents take precedence
over agent guides, help, FAQs, and marketing content. Superseded documents are
not provided and must never be cited. If the evidence does not answer the
question, use decision "clarify" or "escalate" rather than inventing a policy.
"""


class RAGChatbot:
    def __init__(self, client: ChatCompletionClient, model: str, index: KnowledgeBaseIndex):
        self.client = client
        self.model = model
        self.index = index
        self.history: list[dict[str, str]] = []

    def answer(self, question: str, *, limit: int = 15) -> tuple[ChatResult, list[RetrievedEvidence]]:
        if not question.strip():
            raise ValueError("question must not be empty")
        evidence = self.index.retrieve(_retrieval_query(question), limit=limit)
        evidence_text = _format_evidence(evidence)
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}, *self.history]
        messages.append(
            {
                "role": "user",
                "content": f"Customer question:\n{question}\n\nRetrieved evidence:\n{evidence_text}",
            }
        )
        response = self.client.create(model=self.model, messages=messages)
        raw = _response_text(response)
        result = _parse_result(raw)
        self.history.extend(
            [
                {"role": "user", "content": messages[-1]["content"]},
                {"role": "assistant", "content": raw},
            ]
        )
        return result, evidence

    def reset(self) -> None:
        self.history.clear()


def _format_evidence(evidence: list[RetrievedEvidence]) -> str:
    if not evidence:
        return "No matching knowledge-base section was retrieved."
    blocks = []
    for item in evidence:
        metadata = ", ".join(
            f"{key}={value}" for key, value in item.metadata.items()
            if key in {"doc_type", "effective_from", "status", "version"}
        )
        blocks.append(
            f"[{item.section_id}] source={item.document} {item.title} "
            f"({metadata})\n{item.text}"
        )
    return "\n\n".join(blocks)


def _retrieval_query(question: str) -> str:
    """Keep conversational context available to the model, not the ranker."""
    marker = "Customer message:\n"
    if marker not in question:
        return question
    message = question.split(marker, 1)[1]
    return message.split("\n\nVerified order context:", 1)[0].strip()


def evidence_as_dicts(evidence: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    return [
        {
            "section_id": item.section_id,
            "title": item.title,
            "document": item.document,
            "metadata": item.metadata,
            "score": item.score,
        }
        for item in evidence
    ]
