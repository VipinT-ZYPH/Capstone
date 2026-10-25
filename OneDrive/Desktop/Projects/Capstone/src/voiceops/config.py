"""Environment-backed application configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

MODEL_NAME_RE = re.compile(r"^gemini-[a-z0-9][a-z0-9.-]*$", re.IGNORECASE)


@dataclass(frozen=True)
class Settings:
    dataset_root: Path
    reference_date: str = "2026-10-20"
    llm_api_key: str | None = None
    llm_model: str = "gemini-2.5-flash"
    request_delay_seconds: float = 0.0
    gnani_api_key: str | None = None
    gnani_base_url: str = "https://api.vachana.ai"

    @classmethod
    def from_environment(cls, workspace: Path) -> "Settings":
        configured = os.getenv("ZENTROMART_DATASET_DIR")
        dataset_root = Path(configured) if configured else workspace / "Capstone_Project_Dataset_FTTT_2026"
        if not dataset_root.is_absolute():
            dataset_root = workspace / dataset_root
        raw_delay = os.getenv("GEMINI_REQUEST_DELAY_SECONDS", "0")
        try:
            request_delay = max(0.0, float(raw_delay))
        except ValueError as exc:
            raise ValueError("GEMINI_REQUEST_DELAY_SECONDS must be a number") from exc
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        if not MODEL_NAME_RE.fullmatch(model):
            raise ValueError(
                "GEMINI_MODEL must be a Gemini model identifier such as "
                "gemini-2.5-flash; availability is checked by the Gemini API"
            )
        return cls(
            dataset_root=dataset_root,
            reference_date=os.getenv("ZENTROMART_REFERENCE_DATE", "2026-10-20"),
            llm_api_key=os.getenv("GEMINI_API_KEY"),
            llm_model=model,
            request_delay_seconds=request_delay,
            gnani_api_key=os.getenv("GNANI_API_KEY"),
            gnani_base_url=os.getenv("GNANI_BASE_URL", "https://api.vachana.ai").strip(),
        )
