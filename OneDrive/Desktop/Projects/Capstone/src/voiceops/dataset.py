"""Read-only access to the supplied synthetic ZentroMart dataset."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

SECTION_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class DatasetPaths:
    """Locations of the supplied dataset directories."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def knowledge_base(self) -> Path:
        return self.root / "kb"

    @property
    def practice(self) -> Path:
        return self.root / "practice"

    @property
    def validator(self) -> Path:
        return self.root / "validate_dataset.py"

    @classmethod
    def discover(cls, start: Path) -> "DatasetPaths":
        """Find the extracted dataset without assuming the archive nesting."""
        candidates = [start, start / "Capstone_Project_Dataset_FTTT_2026"]
        candidates.extend(
            path.parent
            for path in start.rglob("validate_dataset.py")
            if path.parent.name == "Capstone_Project_Dataset_FTTT_2026"
        )
        for candidate in candidates:
            if (candidate / "data").is_dir() and (candidate / "kb").is_dir():
                return cls(candidate.resolve())
        raise FileNotFoundError(
            f"Could not find an extracted ZentroMart dataset below {start}"
        )


@dataclass(frozen=True)
class KnowledgeDocument:
    path: Path
    metadata: dict[str, str]
    text: str


@dataclass(frozen=True)
class KnowledgeSection:
    document: Path
    metadata: dict[str, str]
    section_id: str
    title: str
    text: str


class DatasetRepository:
    """Load dataset records while preserving identifier columns as strings."""

    def __init__(self, paths: DatasetPaths):
        self.paths = paths

    def read_csv(self, relative_path: str) -> list[dict[str, str]]:
        path = self.paths.root / relative_path
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def documents(self) -> Iterable[KnowledgeDocument]:
        for path in sorted(self.paths.knowledge_base.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            yield KnowledgeDocument(path=path, metadata=_frontmatter(text), text=text)

    def sections(self) -> list[KnowledgeSection]:
        sections: list[KnowledgeSection] = []
        for document in self.documents():
            matches = list(SECTION_RE.finditer(document.text))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(document.text)
                body = document.text[match.end():end].strip()
                sections.append(
                    KnowledgeSection(
                        document=document.path,
                        metadata=document.metadata,
                        section_id=match.group(1),
                        title=match.group(2).strip(),
                        text=body,
                    )
                )
        return sections

    def practice_questions(self) -> list[dict[str, str]]:
        return self.read_csv("practice/questions.csv")

    def practice_conversations(self) -> list[dict[str, str]]:
        return self.read_csv("practice/conversations.csv")


def _frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Knowledge-base document is missing YAML frontmatter")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("Knowledge-base frontmatter is not closed") from exc
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    required = {"doc_id", "title", "doc_type", "effective_from", "status"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Knowledge-base metadata missing: {', '.join(sorted(missing))}")
    return metadata


def business_days_after(start: date, days: int, holidays: set[date]) -> date:
    """Return a date after ``days`` Monday-Saturday business days."""
    if days < 0:
        raise ValueError("days must be non-negative")
    current = start
    remaining = days
    while remaining:
        current += timedelta(days=1)
        if current.weekday() < 6 and current not in holidays:
            remaining -= 1
    return current
