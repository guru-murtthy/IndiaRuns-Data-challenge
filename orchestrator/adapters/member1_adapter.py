from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from ingestion.candidate_loader import load_candidates as _load_candidates
from parser.jd.parser import parse_job_description as _parse_job_description

def parse_job_description(title: str, description: str) -> dict[str, object]:
    """Adapter for JD parsing."""
    return _parse_job_description(title, description)

def load_candidates(candidate_path: Path | None = None) -> list[dict[str, Any]]:
    """Adapter for candidate loading."""
    from utils.constants import CANDIDATES_FILE
    path = candidate_path or CANDIDATES_FILE
    return _load_candidates(path)
