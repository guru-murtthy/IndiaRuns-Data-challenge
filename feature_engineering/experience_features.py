from feature_engineering.company_features import CONSULTING_FIRMS

STARTUP_COMPANIES = {
    "Swiggy", "Razorpay", "CRED", "Zomato", "Flipkart", "Mindtree", "Meesho", "Nykaa", 
    "InMobi", "BYJU'S", "PolicyBazaar", "Ola", "Zoho", "Vedantu", "Paytm", "Unacademy", 
    "PharmEasy", "upGrad", "Freshworks", "PhonePe", "Dream11", "Krutrim", "Sarvam AI", 
    "Glance", "Rephrase.ai", "Aganitha", "Niramai", "Saarthi.ai", "Mad Street Den", 
    "Observe.AI", "Wysa", "Haptik"
}

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
    has_long_stay = False
    durations = []
    
    for job in sorted_jobs:
        company = job.get("company", "")
        title = job.get("title", "").lower()
        duration = job.get("duration_months", 0)
        
        if duration > 0:
            durations.append(duration)
            if duration >= 36:
                has_long_stay = True
        
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
                growth_points -= 1.5 # Moving from product back to service
                
        if prev_title_level > 0:
            # Promotion jump
            if level > prev_title_level:
                growth_points += 2.0
                
        # 3. Stability check (penalize excessive job-hopping: short stints < 12 months)
        if duration < 12 and not job.get("is_current", False):
            stability_penalties += 1
            
        prev_company_tier = tier
        prev_title_level = level
        
    # Tenures check
    avg_duration = sum(durations) / len(durations) if durations else 0
    if len(durations) >= 2 and avg_duration < 18:
        growth_points -= 2.0 # Penalize frequent hoppers
        
    if has_long_stay:
        growth_points += 1.5 # Reward candidates with proven long stays
        
    # Calculate final score
    base_growth_score = min(10.0, max(0.0, growth_points))
    stability_multiplier = max(0.3, 1.0 - (stability_penalties * 0.15))
    
    return base_growth_score * stability_multiplier
