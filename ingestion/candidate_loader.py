import json
from pathlib import Path
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

def _normalize_candidate(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    candidate_id = record.get("candidate_id", f"CAND_{index:07d}").upper()
    profile = record.get("profile", {})
    
    name = profile.get("anonymized_name", f"Candidate {index}")
    title = profile.get("current_title", "Unknown Role")
    location = profile.get("location", "Unknown")
    summary = profile.get("summary", "No summary provided.")
    
    exp_years = profile.get("years_of_experience", 0)
    if not isinstance(exp_years, (int, float)):
        exp_years = 0.0

    raw_skills = record.get("skills", [])
    skills = []
    for s in raw_skills:
        if isinstance(s, dict) and "name" in s:
            skills.append(str(s["name"]).strip().lower())
        elif isinstance(s, str):
            skills.append(s.strip().lower())
    
    skills = sorted(list(set(skills)))
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
        "raw_record": record,
    }

def load_candidates(candidate_path: Path) -> List[Dict[str, Any]]:
    normalized_candidates: List[Dict[str, Any]] = []
    if not candidate_path.exists():
        logger.warning(f"Candidate file not found: {candidate_path}")
        return normalized_candidates
        
    try:
        raw_data = json.loads(candidate_path.read_text(encoding="utf-8"))
        if isinstance(raw_data, list):
            for index, item in enumerate(raw_data, start=1):
                normalized_candidates.append(_normalize_candidate(item, index))
            return normalized_candidates
    except json.JSONDecodeError:
        pass
        
    with candidate_path.open("r", encoding="utf-8") as candidate_file:
        for index, line in enumerate(candidate_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                normalized_candidates.append(_normalize_candidate(json.loads(stripped), index))
            except json.JSONDecodeError:
                pass

    return normalized_candidates
