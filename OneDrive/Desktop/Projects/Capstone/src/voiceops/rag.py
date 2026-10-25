"""Metadata-aware lexical retrieval over ZentroMart KB sections."""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Iterable

from .dataset import KnowledgeSection

# Include Indic combining marks, which Python's ``\w`` does not keep inside
# Devanagari/Tamil words when used alone.
TOKEN_RE = re.compile(r"[^\W_]+[\u0900-\u0DFF\u1E00-\u1EFF]*|[\u0900-\u0DFF]+", re.UNICODE)
STOP_WORDS = {
    "a", "about", "and", "are", "can", "do", "for", "how", "i", "is", "it",
    "me", "my", "of", "on", "or", "the", "to", "what", "when", "where",
    "which", "with", "you", "your",
}
SYNONYM_GROUPS = (
    ("return", "returns", "returned", "send", "back", "लौटाने", "वापस"),
    ("refund", "refunds", "money", "पैसे", "रिफंड"),
    ("pickup", "pickups", "collection", "collect", "उठाने"),
    ("damage", "damaged", "broken", "crack", "torn", "खराब", "टूटा"),
    ("warranty", "warranties", "assured", "claim", "वारंटी"),
    ("days", "day", "window", "long", "time", "期限", "दिन", "कितने"),
    ("cancel", "cancellation", "कैंसिल"),
    ("shipped", "shipping", "dispatch", "शिप"),
    ("opened", "open", "tried", "used", "खोला"),
    ("clothing", "clothes", "fashion", "kurta", "jeans", "वस्त्र"),
    ("beauty", "lipstick", "perfume", "fragrance", "cosmetic"),
    ("books", "book", "notebook", "stationery"),
    ("electronics", "laptop", "smartphone", "smartwatch", "powerbank"),
    ("swimsuit", "briefs", "innerwear", "lingerie", "swimwear"),
    ("ship", "shipped", "shipping", "dispatch", "शिप"),
    ("cash", "cod", "cash-on-delivery", "bank", "transfer"),
    ("sorted", "resolve", "help", "human", "escalate", "colleague"),
    ("late", "delay", "delayed", "overdue", "express", "credit"),
    (
        "phone", "number", "contact", "privacy", "personal", "details",
        "address", "person", "customer", "share", "another", "full",
        "ordered", "building",
    ),
    ("opened", "open", "tried", "used", "unused", "condition", "seal", "खोला"),
)
DOC_TYPE_PRIORITY = {
    "policy": 4,
    "agent_guide": 3,
    "help": 2,
    "faq": 1,
    "marketing": 0,
    "meta": 0,
}


@dataclass(frozen=True)
class RetrievedEvidence:
    section_id: str
    title: str
    text: str
    document: str
    metadata: dict[str, str]
    score: float


class KnowledgeBaseIndex:
    """Index parsed sections and retrieve only answer-authoritative evidence."""

    def __init__(self, sections: Iterable[KnowledgeSection]):
        self.sections = [
            section for section in sections
            if section.metadata.get("status", "").casefold() != "superseded"
        ]
        self._tokens = {
            section.section_id: _tokens(
                f"{section.title} {section.metadata.get('title', '')} "
                f"{section.document.stem} {section.text}"
            )
            for section in self.sections
        }
        self._title_tokens = {
            section.section_id: _tokens(
                f"{section.title} {section.metadata.get('title', '')}"
            )
            for section in self.sections
        }
        self._by_id = {section.section_id: section for section in self.sections}
        self._document_frequency = {
            token: sum(token in tokens for tokens in self._tokens.values())
            for tokens in self._tokens.values()
            for token in tokens
        }
        self._document_count = max(len(self.sections), 1)

    def retrieve(self, query: str, *, limit: int = 15) -> list[RetrievedEvidence]:
        if not query.strip():
            raise ValueError("query must not be empty")
        query_tokens = _tokens(query)
        governance_ids = self._governance_context(query)
        scored: list[RetrievedEvidence] = []
        for section in self.sections:
            section_tokens = self._tokens[section.section_id]
            overlap = query_tokens & section_tokens
            if not overlap and section.section_id not in governance_ids:
                continue
            original_query_tokens = _raw_tokens(query)
            exact_overlap = original_query_tokens & _raw_tokens(
                f"{section.title} {section.text}"
            )
            title_overlap = _raw_tokens(query) & _raw_tokens(
                f"{section.title} {section.metadata.get('title', '')}"
            )
            phrase_overlap = _phrase_overlap(
                query,
                f"{section.title} {section.text}",
            )
            weighted_overlap = sum(self._idf(token) for token in overlap)
            weighted_exact = sum(self._idf(token) for token in exact_overlap)
            query_weight = sum(
                self._idf(token)
                for token in query_tokens
                if token in self._document_frequency
            )
            score = (weighted_exact * 2 + weighted_overlap) / max(query_weight * 2, 1)
            score += len(title_overlap) * 0.2
            score += phrase_overlap * 0.4
            if section.section_id in governance_ids:
                score += 0.2
            score += _metadata_boost(section)
            scored.append(
                RetrievedEvidence(
                    section_id=section.section_id,
                    title=section.title,
                    text=section.text,
                    document=str(section.document),
                    metadata=dict(section.metadata),
                    score=round(score, 6),
                )
            )
        scored.sort(key=lambda item: (-item.score, item.section_id))
        selected = [item for item in scored if item.section_id in governance_ids]
        selected.extend(item for item in scored if item.section_id not in governance_ids)
        return selected[:limit]

    def _idf(self, token: str) -> float:
        frequency = self._document_frequency.get(token, 0)
        return math.log((self._document_count + 1) / (frequency + 1)) + 1

    def _governance_context(self, query: str) -> set[str]:
        query_tokens = _raw_tokens(query)
        ids: set[str] = set()
        precedence_terms = {
            "conflict", "current", "document", "override", "policy",
            "refund", "refunds", "return", "returns", "superseded", "version",
        }
        if query_tokens & precedence_terms:
            ids.update(
                section.section_id
                for section in self.sections
                if section.metadata.get("doc_type") == "meta"
                and "which document wins" in section.title.casefold()
            )
        timeline_terms = {
            "business", "calendar", "date", "deadline", "holiday",
            "timeline", "started", "october", "november", "2026",
        }
        if query_tokens & timeline_terms:
            ids.update(
                section.section_id
                for section in self.sections
                if section.metadata.get("doc_type") == "meta"
                and "business days" in section.title.casefold()
            )
        sensitive_terms = {
            "address", "building", "contact", "email", "number", "person",
            "phone", "street",
        }
        if query_tokens & sensitive_terms:
            ids.update(
                section.section_id
                for section in self.sections
                if section.metadata.get("doc_type") == "agent_guide"
                and (
                    "never read out" in section.title.casefold()
                    or "things you must never do" in section.title.casefold()
                )
            )
        if _is_low_information_opening(query):
            ids.update(
                section.section_id
                for section in self.sections
                if section.metadata.get("doc_type") == "agent_guide"
                and section.title.casefold() == "opening"
            )
        handover_terms = {
            "approve", "approval", "colleague", "connect", "escalate",
            "escalation", "human", "insaan", "ticket", "तीसरी", "मानव",
        }
        if query_tokens & handover_terms:
            ids.update(
                section.section_id
                for section in self.sections
                if section.metadata.get("doc_type") == "agent_guide"
                and (
                    "hand over" in section.title.casefold()
                    or "must never do" in section.title.casefold()
                )
            )
        return ids


def _tokens(value: str) -> set[str]:
    tokens = set()
    for token in _raw_tokens(value):
        if token in STOP_WORDS or len(token) <= 1:
            continue
        tokens.add(token)
        if token.endswith("ies") and len(token) > 4:
            tokens.add(token[:-3] + "y")
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            tokens.add(token[:-1])
    expanded = set(tokens)
    for group in SYNONYM_GROUPS:
        if tokens.intersection(group):
            expanded.update(group)
    return expanded


def _raw_tokens(value: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(value.casefold())
        if token not in STOP_WORDS and len(token) > 1
    }


def _phrase_overlap(query: str, text: str) -> int:
    query_tokens = _raw_token_sequence(query)
    text_phrases = set(zip(_raw_token_sequence(text), _raw_token_sequence(text)[1:]))
    return sum(
        pair in text_phrases
        for pair in zip(query_tokens, query_tokens[1:])
    )


def _raw_token_sequence(value: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(value.casefold())
        if token not in STOP_WORDS and len(token) > 1
    ]


def _metadata_boost(section: KnowledgeSection) -> float:
    boost = DOC_TYPE_PRIORITY.get(section.metadata.get("doc_type", ""), 0) * 0.02
    if section.metadata.get("status", "").casefold() == "current":
        boost += 0.05
    return boost


def _is_low_information_opening(query: str) -> bool:
    tokens = _raw_tokens(query)
    opening_terms = {
        "hello", "hi", "hey", "help", "order", "नमस्ते", "ऑर्डर",
    }
    return (
        ("help" in tokens and "order" in tokens)
        or ("नमस्ते" in query and ("मदद" in query or "ऑर्डर" in query))
        or ("hello" in tokens and len(tokens) <= 8)
        or ("hi" in tokens and len(tokens) <= 8)
    )
