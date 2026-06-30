from typing import Dict, Any, List

def generate_reasoning(candidate: Dict[str, Any], raw_record: Dict[str, Any], overlap: List[str], jd: Dict[str, Any], is_honeypot: bool) -> str:
    """
    Generate honest, candidate-specific reasoning based on actual profile data.
    """
    if is_honeypot:
        return (
            f"{candidate.get('title', 'Candidate')} with "
            f"{candidate.get('experience_years', 0):.1f} yrs experience; "
            "profile contains inconsistencies suggesting data quality issues — ranked low."
        )

    signals = raw_record.get("redrob_signals", {})
    career = raw_record.get("career_history", [])
    
    parts = []
    title = candidate.get("title", "")
    yrs = float(candidate.get("experience_years", 0))
    parts.append(f"{title} with {yrs:.1f} yrs")

    if career:
        recent = career[0]
        co = recent.get("company", "")
        if co:
            parts.append(f"currently at {co}")

    if overlap:
        skill_str = ", ".join(overlap[:3])
        parts.append(f"matches on {skill_str}")
    else:
        parts.append("adjacent skills")

    notice = int(signals.get("notice_period_days", 0))
    open_to_work = signals.get("open_to_work_flag", False)
    if open_to_work and notice <= 30:
        parts.append("available immediately")
    elif notice > 90:
        parts.append(f"notice {notice}d — potential delay")

    loc = candidate.get("location", "")
    if loc:
        parts.append(loc)

    return "; ".join(parts) + "."
