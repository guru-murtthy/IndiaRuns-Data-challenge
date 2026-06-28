import numpy as np
from feature_engineering.company_features import is_consulting_only_candidate
from feature_engineering.skill_features import calculate_skill_evidence_score
from feature_engineering.experience_features import calculate_career_growth_score

def extract_features(cand, authenticity_score_fn):
    """
    Extracts features from a candidate record to be used by the LightGBM model.
    """
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    is_consulting_only = is_consulting_only_candidate(cand)
    
    expected_salary = signals.get("expected_salary_range_inr_lpa", {})
    salary_min = expected_salary.get("min", 15.0)
    salary_max = expected_salary.get("max", 30.0)
    
    skill_evidence = calculate_skill_evidence_score(cand)
    career_growth = calculate_career_growth_score(cand)
    authenticity = authenticity_score_fn(cand)
    
    features = {
        "candidate_id": cand.get("candidate_id"),
        "years_of_experience": float(profile.get("years_of_experience", 0.0)),
        "skill_count": float(len(skills)),
        "skill_evidence_score": skill_evidence,
        "career_growth_score": career_growth,
        "resume_authenticity_score": authenticity,
        "profile_completeness_score": float(signals.get("profile_completeness_score", 0.0)),
        "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0.0)),
        "avg_response_time_hours": float(signals.get("avg_response_time_hours", 24.0)),
        "connection_count": float(signals.get("connection_count", 0.0)),
        "endorsements_received": float(signals.get("endorsements_received", 0.0)),
        "notice_period_days": float(signals.get("notice_period_days", 30.0)),
        "salary_min": float(salary_min),
        "salary_max": float(salary_max),
        "github_activity_score": float(signals.get("github_activity_score", -1.0)),
        "search_appearance_30d": float(signals.get("search_appearance_30d", 0.0)),
        "saved_by_recruiters_30d": float(signals.get("saved_by_recruiters_30d", 0.0)),
        "interview_completion_rate": float(signals.get("interview_completion_rate", 1.0)),
        "offer_acceptance_rate": float(signals.get("offer_acceptance_rate", -1.0)),
        "is_consulting_only": is_consulting_only
    }
    return features
