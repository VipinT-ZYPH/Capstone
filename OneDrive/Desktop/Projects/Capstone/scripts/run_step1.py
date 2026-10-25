"""Answer all supplied practice questions with the prompt-only chatbot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from voiceops.chatbot import GeminiChatClient, ZentroMartChatbot
from voiceops.config import Settings
from voiceops.dataset import DatasetPaths, DatasetRepository
from voiceops.evaluation import evaluate_questions, records_as_dicts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/step1_results.json"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    load_dotenv(workspace / ".env")
    settings = Settings.from_environment(workspace)
    if not settings.llm_api_key:
        raise SystemExit("GEMINI_API_KEY is not configured; copy .env.example to .env")
    paths = DatasetPaths.discover(workspace)
    repository = DatasetRepository(paths)
    chatbot = ZentroMartChatbot(
        client=GeminiChatClient(settings.llm_api_key),
        model=settings.llm_model,
    )
    records = evaluate_questions(
        repository.practice_questions(),
        chatbot,
        request_delay_seconds=settings.request_delay_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "prompt-only",
        "model": settings.llm_model,
        "reference_date": settings.reference_date,
        "results": records_as_dicts(records),
        "summary": {
            "total": len(records),
            "scored": sum(record.scored for record in records),
            "errors": sum(not record.scored for record in records),
            "correct": sum(record.correct for record in records if record.scored),
            "decision_correct": sum(
                record.decision_correct for record in records if record.scored
            ),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"]))
    print(f"Saved {len(records)} results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
