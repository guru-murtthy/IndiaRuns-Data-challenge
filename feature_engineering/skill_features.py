def calculate_skill_evidence_score(candidate):
    """
    Computes a reliability score for the candidate's skill list.
    Weighs proficiency against duration and endorsements to detect keyword stuffing.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0
        
    total_score = 0.0
    prof_weights = {"beginner": 1.0, "intermediate": 2.0, "advanced": 3.0, "expert": 4.0}
    
    for s in skills:
        prof = s.get("proficiency", "beginner").lower()
        weight = prof_weights.get(prof, 1.0)
        endorsements = s.get("endorsements", 0)
        duration = s.get("duration_months", 0)
        
        # Evidence check:
        # A candidate who lists a skill but has 0 duration and 0 endorsements gets 0 evidence multiplier
        if duration == 0 and endorsements == 0:
            evidence_factor = 0.05
        else:
            # Combined evidence of duration and endorsements
            duration_factor = min(1.0, duration / 24.0) # Max weight at 2 years
            endorsements_factor = min(1.0, endorsements / 15.0) # Max weight at 15 endorsements
            evidence_factor = duration_factor * 0.7 + endorsements_factor * 0.3
            
        skill_score = weight * evidence_factor
        total_score += skill_score
        
    # Average score across all listed skills
    return total_score / len(skills)
