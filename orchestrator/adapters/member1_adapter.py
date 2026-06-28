from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SKILL_KEYWORDS = {
    "machine learning", "deep learning", "python", "faiss", "fastapi", "docker", 
    "lightgbm", "catboost", "embeddings", "retrieval", "ranking", "csv", "nlp", 
    "analytics", "react", "sql", "aws", "gcp", "azure", "kubernetes"
}

RAW_CANDIDATE_PATH = Path("data/raw/candidates.jsonl")
RAW_JD_PATH = Path("data/raw/jd.json")

def parse_job_description(title: str, description: str) -> dict[str, object]:
    lowered = description.lower()
    skills = sorted({skill for skill in SKILL_KEYWORDS if skill in lowered})
    years_match = re.search(r"(\d+)\+?\s+years", lowered)
    min_experience = int(years_match.group(1)) if years_match else 2

    return {
        "title": title,
        "description": description,
        "required_skills": skills,
        "min_experience_years": min_experience,
    }

def _normalize_candidate(record: dict[str, Any], index: int) -> dict[str, Any]:
    # Handle the official candidate schema
    candidate_id = record.get("candidate_id", f"CAND_{index:07d}").upper()
    
    profile = record.get("profile", {})
    name = profile.get("anonymized_name", f"Candidate {index}")
    title = profile.get("current_title", "Unknown Role")
    location = profile.get("location", "Unknown")
    summary = profile.get("summary", "No summary provided.")
    
    # Parse years of experience properly
    exp_years = profile.get("years_of_experience", 0)
    if not isinstance(exp_years, (int, float)):
        exp_years = 0.0

    # Parse skills according to schema (list of dicts with 'name' and 'proficiency')
    raw_skills = record.get("skills", [])
    skills = []
    for s in raw_skills:
        if isinstance(s, dict) and "name" in s:
            skills.append(str(s["name"]).strip().lower())
        elif isinstance(s, str):
            skills.append(s.strip().lower())
    
    # Dedup and sort
    skills = sorted(list(set(skills)))

    # Parse signals
    signals = record.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate", 0.5)

    return {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "title": title,
        "location": location,
        "skills": skills,
        "experience_years": float(exp_years),
        "summary": summary,
        "response_rate": float(response_rate),
        "raw_record": record, # Keep original record for ranking features if needed
    }

def load_candidates(candidate_path: Path | None = None) -> list[dict[str, Any]]:
    path = candidate_path or RAW_CANDIDATE_PATH
    normalized_candidates: list[dict[str, Any]] = []

    if path.exists():
        try:
            # Attempt to read as a single JSON array (e.g. sample lists)
            raw_data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw_data, list):
                for index, item in enumerate(raw_data, start=1):
                    normalized_candidates.append(_normalize_candidate(item, index))
                return normalized_candidates
        except json.JSONDecodeError:
            pass # Fallback to JSONL
            
        # Parse as JSONL (handles the real 100K candidates file)
        with path.open("r", encoding="utf-8") as candidate_file:
            for index, line in enumerate(candidate_file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    normalized_candidates.append(_normalize_candidate(json.loads(stripped), index))
                except json.JSONDecodeError:
                    pass

    return normalized_candidates
