"""Validate the supplied dataset and the Phase 1 read-only data foundation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from voiceops.dataset import DatasetPaths, DatasetRepository


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    paths = DatasetPaths.discover(workspace)
    repository = DatasetRepository(paths)
    sections = repository.sections()
    questions = repository.practice_questions()
    conversations = repository.practice_conversations()
    if len(sections) == 0 or len(questions) != 52 or len(conversations) != 96:
        raise RuntimeError("Phase 1 dataset counts do not match the supplied schema")
    result = subprocess.run(
        [sys.executable, str(paths.validator)],
        cwd=paths.root,
        check=False,
    )
    if result.returncode:
        return result.returncode
    print(f"Phase 1 loaded {len(sections)} KB sections from {paths.root}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
