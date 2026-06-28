from .timeline_validation import validate_candidate_timeline, parse_date
import datetime

CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini", "HCL", "Tech Mahindra", "Mphasis"
}

STARTUP_COMPANIES = {
    "Swiggy", "Razorpay", "CRED", "Zomato", "Flipkart", "Mindtree", "Meesho", "Nykaa", 
    "InMobi", "BYJU'S", "PolicyBazaar", "Ola", "Zoho", "Vedantu", "Paytm", "Unacademy", 
    "PharmEasy", "upGrad", "Freshworks", "PhonePe", "Dream11", "Krutrim", "Sarvam AI", 
    "Glance", "Rephrase.ai", "Aganitha", "Niramai", "Saarthi.ai", "Mad Street Den", 
    "Observe.AI", "Wysa", "Haptik"
}

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

def calculate_career_growth_score(candidate):
    """
    Evaluates career trajectory based on title growth, company types, and job stability.
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.0
        
    # Sort chronologically to scan career path
    sorted_jobs = sorted(
        [job for job in history if job.get("start_date")],
        key=lambda x: x.get("start_date")
    )
    
    growth_points = 0.0
    prev_company_tier = 0
    prev_title_level = 0
    
    title_hierarchy = {
        "intern": 1, "junior": 2, "engineer": 3, "developer": 3, 
        "senior": 4, "lead": 5, "principal": 5, "manager": 6, "director": 7
    }
    
    stability_penalties = 0
    
    for job in sorted_jobs:
        company = job.get("company", "")
        title = job.get("title", "").lower()
        duration = job.get("duration_months", 0)
        
        # 1. Company tier evaluation (jumping from service firms to product startups)
        if company in CONSULTING_FIRMS:
            tier = 1
        elif company in STARTUP_COMPANIES:
            tier = 3
        else:
            tier = 2
            
        # 2. Title level evaluation
        level = 3 # default
        for role_keyword, lv in title_hierarchy.items():
            if role_keyword in title:
                level = max(level, lv)
                
        # Calculate growth jumps
        if prev_company_tier > 0:
            # Service company to product company jump
            if tier > prev_company_tier:
                growth_points += 2.5
            elif tier < prev_company_tier:
                growth_points -= 1.0 # Moving from product back to service
                
        if prev_title_level > 0:
            # Promotion jump
            if level > prev_title_level:
                growth_points += 2.0
                
        # 3. Stability check (penalize excessive job-hopping: short stints < 12 months)
        if duration < 12 and not job.get("is_current", False):
            stability_penalties += 1
            
        prev_company_tier = tier
        prev_title_level = level
        
    # Calculate final score
    base_growth_score = min(10.0, max(0.0, growth_points))
    stability_multiplier = max(0.4, 1.0 - (stability_penalties * 0.15))
    
    return base_growth_score * stability_multiplier

def calculate_resume_authenticity_score(candidate):
    """
    Evaluates profile authenticity. Combines timeline validations with platform verification signals.
    """
    is_valid, _, anomalies = validate_candidate_timeline(candidate)
    
    # If the profile fails strict timeline/honeypot checks, it is flagged as inauthentic (score 0.0)
    if not is_valid:
        return 0.0
        
    score = 100.0
    signals = candidate.get("redrob_signals", {})
    
    # Platform verifications
    if not signals.get("verified_email", False):
        score -= 10.0
    if not signals.get("verified_phone", False):
        score -= 10.0
    if not signals.get("linkedin_connected", False):
        score -= 5.0
        
    # GitHub activity check
    github_score = signals.get("github_activity_score", -1)
    if github_score == -1:
        score -= 10.0 # No GitHub connected is a negative signal for an ML Engineer role
        
    # Account activity check
    # Signup date to last active date must be reasonable
    signup_s = signals.get("signup_date")
    last_active_s = signals.get("last_active_date")
    signup_d = parse_date(signup_s)
    last_active_d = parse_date(last_active_s)
    
    if signup_d and last_active_d:
        if signup_d > last_active_d:
            score -= 20.0 # signup after last active is a synthetic data anomaly
            
    # Minor timeline inconsistencies
    for job in candidate.get("career_history", []):
        start_s = job.get("start_date")
        end_s = job.get("end_date")
        dur_reported = job.get("duration_months")
        
        start_d = parse_date(start_s)
        end_d = parse_date(end_s) if end_s else datetime.date(2026, 6, 28)
        
        if start_d and end_d:
            delta_days = (end_d - start_d).days
            dur_calc = round(delta_days / 30.4375)
            if dur_reported is not None and abs(dur_calc - dur_reported) > 2:
                score -= 3.0 # subtract for minor duration discrepancies
                
    return max(0.0, score)
