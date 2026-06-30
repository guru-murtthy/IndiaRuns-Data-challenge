from __future__ import annotations
from validation.honeypot import detect_honeypot
from reasoning.reasoning_engine import generate_reasoning

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
        
        is_honeypot = detect_honeypot(candidate, raw)

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

        reason = generate_reasoning(candidate, raw, overlap, jd, is_honeypot)

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
