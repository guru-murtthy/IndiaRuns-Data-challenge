from typing import Dict, Any

def detect_honeypot(candidate: Dict[str, Any], raw_record: Dict[str, Any]) -> bool:
    """
    Basic honeypot detection using logic from the submission_spec.
    Honeypots have subtly impossible profiles (e.g. 8 years at a 3-year old company).
    """
    career_history = raw_record.get("career_history", [])
    for job in career_history:
        duration = int(job.get("duration_months", 0))
        if duration > 600: # Very long tenure at company (>50 years)
            return True

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
