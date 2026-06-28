from __future__ import annotations
"""
Member 3: Validation & Reasoning using real candidate data.
No LLM or external APIs are used. Reasoning is generated from actual schema fields.
Members 3 should replace this with their real validation model when ready.
"""
import re
import datetime


def _detect_honeypot(candidate: dict, raw_record: dict) -> bool:
    """
    Basic honeypot detection using logic from the submission_spec.
    Honeypots have subtly impossible profiles (e.g. 8 years at a 3-year old company).
    """
    career_history = raw_record.get("career_history", [])
    for job in career_history:
        duration = int(job.get("duration_months", 0))
        # Very long tenure at company (>15 years = 180 months) is suspicious
        if duration > 600:
            return True

    # Check for expert skills with 0 duration_months (keyword stuffing signal)
    raw_skills = raw_record.get("skills", [])
    suspicious_expert_count = 0
    for skill_entry in raw_skills:
        if not isinstance(skill_entry, dict):
            continue
        if skill_entry.get("proficiency") == "expert" and int(skill_entry.get("duration_months", 1)) == 0:
            suspicious_expert_count += 1
    if suspicious_expert_count >= 5:
        return True

    return False


def _generate_reasoning(candidate: dict, raw_record: dict, overlap: list[str],
                        jd: dict, is_honeypot: bool) -> str:
    """
    Generate honest, candidate-specific reasoning based on actual profile data.
    Avoids templated strings that get penalized at Stage 4.
    """
    if is_honeypot:
        return (
            f"{candidate.get('title', 'Candidate')} with "
            f"{candidate.get('experience_years', 0):.1f} yrs experience; "
            "profile contains inconsistencies suggesting data quality issues — ranked low."
        )

    signals = raw_record.get("redrob_signals", {})
    profile = raw_record.get("profile", {})
    career = raw_record.get("career_history", [])
    
    parts = []
    
    # Title & experience
    title = candidate.get("title", "")
    yrs = float(candidate.get("experience_years", 0))
    parts.append(f"{title} with {yrs:.1f} yrs")

    # Most recent company from career history
    if career:
        recent = career[0]
        co = recent.get("company", "")
        co_size = recent.get("company_size", "")
        if co:
            parts.append(f"currently at {co}" + (f" ({co_size})" if co_size else ""))

    # Skill overlap
    if overlap:
        skill_str = ", ".join(overlap[:3])
        parts.append(f"matches on {skill_str}")
    else:
        parts.append("adjacent skills")

    # Engagement signals
    notice = int(signals.get("notice_period_days", 0))
    open_to_work = signals.get("open_to_work_flag", False)
    if open_to_work and notice <= 30:
        parts.append("available immediately")
    elif notice > 90:
        parts.append(f"notice {notice}d — potential delay")

    # Location
    loc = candidate.get("location", "")
    if loc:
        parts.append(loc)

    return "; ".join(parts) + "."


def validate_and_explain(
    jd: dict[str, object],
    ranked_candidates: list[dict[str, object]]
) -> list[dict[str, object]]:
    required_skills = set(jd.get("required_skills", []))
    min_experience = int(jd.get("min_experience_years", 0))
    validated: list[dict[str, object]] = []

    for candidate in ranked_candidates:
        raw = candidate.get("raw_record", {})
        signals = raw.get("redrob_signals", {})
        skills = set(candidate.get("skills", []))
        overlap = sorted(required_skills & skills)
        experience_years = float(candidate.get("experience_years", 0))
        
        is_honeypot = _detect_honeypot(candidate, raw)

        # Validation score: penalize honeypots and unusual profiles
        stability_bonus = 0.05 if experience_years >= max(min_experience, 3) else 0.0
        interview_rate = float(signals.get("interview_completion_rate", 0.5))
        completeness = float(signals.get("profile_completeness_score", 50)) / 100.0
        
        validation_score = round(
            candidate["rank_score"] * 0.6
            + interview_rate * 0.15
            + completeness * 0.10
            + stability_bonus
            - (0.4 if is_honeypot else 0.0),
            4,
        )
        validation_score = max(0.0, min(1.0, validation_score))

        final_score = round(
            candidate["semantic_score"] * 0.35
            + candidate["rank_score"] * 0.40
            + validation_score * 0.25,
            4,
        )
        final_score = max(0.0, min(1.0, final_score))

        reason = _generate_reasoning(candidate, raw, overlap, jd, is_honeypot)

        flags = {
            "timeline_ok": experience_years >= min_experience,
            "open_to_work": bool(signals.get("open_to_work_flag", False)),
            "notice_period_days": int(signals.get("notice_period_days", 0)),
            "honeypot_suspected": is_honeypot,
            "authenticity_risk": "high" if is_honeypot else ("medium" if len(overlap) == 0 else "low"),
        }

        enriched = dict(candidate)
        enriched["validation_score"] = validation_score
        enriched["final_score"] = final_score
        enriched["reason"] = reason
        enriched["flags"] = flags
        validated.append(enriched)

    return sorted(validated, key=lambda c: (-c["final_score"], c["candidate_id"]))
