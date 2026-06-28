from __future__ import annotations
"""
Member 2: Heuristic Ranker using signals from the official candidate_schema.json.
This is a CPU-only, offline ranker. No external APIs are used.
Members 2 should replace the heuristic logic here with FAISS + LTR when ready.
"""


def _title_match_score(candidate_title: str, jd_title: str) -> float:
    """Boost candidates whose current title aligns with the target role."""
    title_lower = candidate_title.lower()
    jd_lower = jd_title.lower()
    ai_ml_tokens = {"ai", "ml", "machine learning", "data scientist", "data engineer",
                    "nlp", "deep learning", "search", "retrieval", "ranking", "llm", "research"}
    target_tokens = set(jd_lower.split())
    if any(tok in title_lower for tok in ai_ml_tokens & target_tokens):
        return 1.0
    if any(tok in title_lower for tok in ai_ml_tokens):
        return 0.6
    if any(tok in title_lower for tok in ("software engineer", "backend", "developer")):
        return 0.3
    return 0.1


def _skill_score(candidate_skills: list[str], required_skills: list[str],
                 raw_record: dict) -> float:
    """
    Score based on skill overlap AND skill depth (proficiency + endorsements + duration).
    Uses the raw_record's nested skills array to access depth signals.
    """
    required_set = set(required_skills)
    if not required_set:
        return 0.5  # No requirements means no filter

    # Simple overlap fraction
    candidate_skill_set = set(candidate_skills)
    overlap_count = len(required_set & candidate_skill_set)
    base_score = overlap_count / len(required_set)

    # Bonus from skill depth via raw_record's skills list
    depth_bonus = 0.0
    raw_skills = raw_record.get("skills", [])
    proficiency_weights = {"expert": 0.04, "advanced": 0.02, "intermediate": 0.01, "beginner": 0.0}
    for skill_entry in raw_skills:
        if not isinstance(skill_entry, dict):
            continue
        skill_name = str(skill_entry.get("name", "")).lower()
        if skill_name in required_set:
            prof = skill_entry.get("proficiency", "beginner")
            endorsements = min(int(skill_entry.get("endorsements", 0)), 20)
            depth_bonus += proficiency_weights.get(prof, 0.0) + endorsements * 0.001
    depth_bonus = min(depth_bonus, 0.15)
    return min(base_score + depth_bonus, 1.0)


def _engagement_score(raw_record: dict) -> float:
    """
    Use redrob_signals to measure how active and open the candidate is.
    Active candidates convert better in real recruiting pipelines.
    """
    signals = raw_record.get("redrob_signals", {})
    score = 0.0

    # Open to work is a strong positive
    if signals.get("open_to_work_flag", False):
        score += 0.3
    # Response rate (0-1)
    score += float(signals.get("recruiter_response_rate", 0.0)) * 0.2
    # GitHub activity (0-100 scale -> normalize to 0-1)
    github = float(signals.get("github_activity_score", -1))
    if github >= 0:
        score += (github / 100.0) * 0.15
    # Profile completeness
    completeness = float(signals.get("profile_completeness_score", 0.0))
    score += (completeness / 100.0) * 0.1
    # Saved by recruiters recently (capped signal)
    saved = min(int(signals.get("saved_by_recruiters_30d", 0)), 10)
    score += saved * 0.01
    # Interview completion rate
    score += float(signals.get("interview_completion_rate", 0.0)) * 0.1

    return min(score, 1.0)


def _experience_score(experience_years: float, min_experience: int) -> float:
    """Reward meeting the minimum and penalize far over/under."""
    if min_experience <= 0:
        return 0.8
    ratio = experience_years / min_experience
    if ratio < 0.5:
        return 0.1
    if ratio < 1.0:
        return 0.5 + ratio * 0.4
    if ratio <= 2.5:
        return 1.0  # Sweet spot
    return max(1.0 - (ratio - 2.5) * 0.05, 0.6)  # Slight penalty for extreme over-qualified


def rank_candidates(
    jd: dict[str, object],
    candidates: list[dict[str, object]],
    top_k: int
) -> list[dict[str, object]]:
    """
    Offline heuristic ranker using signals from the official candidate_schema.json.
    Weights:
      - Skill match (with depth):  45%
      - Experience fit:            20%
      - Title alignment:           20%
      - Engagement signals:        15%
    """
    required_skills: list[str] = list(jd.get("required_skills", []))
    min_experience: int = int(jd.get("min_experience_years", 0))
    jd_title: str = str(jd.get("title", ""))

    scored: list[dict[str, object]] = []
    for candidate in candidates:
        raw = candidate.get("raw_record", {})

        skill_sc = _skill_score(candidate.get("skills", []), required_skills, raw)
        exp_sc = _experience_score(float(candidate.get("experience_years", 0)), min_experience)
        title_sc = _title_match_score(str(candidate.get("title", "")), jd_title)
        engagement_sc = _engagement_score(raw)

        semantic_score = round(skill_sc, 4)
        rank_score = round(
            skill_sc * 0.45 + exp_sc * 0.20 + title_sc * 0.20 + engagement_sc * 0.15, 4
        )

        enriched = dict(candidate)
        enriched["semantic_score"] = semantic_score
        enriched["rank_score"] = rank_score
        scored.append(enriched)

    return sorted(
        scored,
        key=lambda c: (c["rank_score"], c["semantic_score"], c["experience_years"], c["candidate_id"]),
        reverse=True,
    )[:top_k]
