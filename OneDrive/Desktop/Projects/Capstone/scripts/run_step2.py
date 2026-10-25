"""Compare prompt-only and RAG answers over all supplied practice questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from voiceops.chatbot import GeminiChatClient, ZentroMartChatbot
from voiceops.config import Settings
from voiceops.dataset import DatasetPaths, DatasetRepository
from voiceops.evaluation import (
    evaluate_questions,
    evaluate_rag_questions,
    records_as_dicts,
)
from voiceops.rag import KnowledgeBaseIndex
from voiceops.rag_chatbot import RAGChatbot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/step2_results.json"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    load_dotenv(workspace / ".env")
    settings = Settings.from_environment(workspace)
    if not settings.llm_api_key:
        raise SystemExit("GEMINI_API_KEY is not configured; copy .env.example to .env")

    repository = DatasetRepository(DatasetPaths.discover(workspace))
    questions = repository.practice_questions()
    prompt_bot = ZentroMartChatbot(GeminiChatClient(settings.llm_api_key), settings.llm_model)
    rag_bot = RAGChatbot(
        GeminiChatClient(settings.llm_api_key),
        settings.llm_model,
        KnowledgeBaseIndex(repository.sections()),
    )
    baseline = evaluate_questions(
        questions,
        prompt_bot,
        request_delay_seconds=settings.request_delay_seconds,
    )
    rag = evaluate_rag_questions(
        questions,
        rag_bot,
        request_delay_seconds=settings.request_delay_seconds,
    )
    payload = {
        "mode": "prompt-only-vs-rag",
        "model": settings.llm_model,
        "reference_date": settings.reference_date,
        "baseline": {
            "results": records_as_dicts(baseline),
            "summary": _summary(baseline),
        },
        "rag": {
            "results": records_as_dicts(rag),
            "summary": _summary(rag),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"baseline": payload["baseline"]["summary"], "rag": payload["rag"]["summary"]}))
    print(f"Saved {len(questions)} baseline and RAG results to {args.output}")
    return 0


def _summary(records):
    return {
        "total": len(records),
        "scored": sum(record.scored for record in records),
        "errors": sum(not record.scored for record in records),
        "correct": sum(record.correct for record in records if record.scored),
        "decision_correct": sum(
            record.decision_correct for record in records if record.scored
        ),
        "evidence_recall": sum(
            record.evidence_recall is True for record in records
            if record.evidence_recall is not None
        ),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
