import datetime
from ingestion.validators import parse_date

# Founding years of well-known startups in the dataset company list
STARTUP_FOUNDING = {
    "Krutrim": 2023,
    "Sarvam AI": 2023,
    "Rephrase.ai": 2019,
    "CRED": 2018,
    "Razorpay": 2014,
    "Swiggy": 2014,
    "Zomato": 2008,
    "Ola": 2010,
    "PhonePe": 2015,
    "Meesho": 2015,
    "Nykaa": 2012,
    "InMobi": 2007,
    "BYJU'S": 2011,
    "PolicyBazaar": 2008,
    "Zoho": 1996,
    "Vedantu": 2011,
    "Paytm": 2010,
    "Unacademy": 2015,
    "PharmEasy": 2015,
    "upGrad": 2015,
    "Freshworks": 2010,
    "Dream11": 2008,
    "Glance": 2019,
    "Aganitha": 2017,
    "Niramai": 2016,
    "Saarthi.ai": 2017,
    "Mad Street Den": 2013,
    "Observe.AI": 2017,
    "Wysa": 2015,
    "Haptik": 2013,
    "Verloop.io": 2016,
    "Yellow.ai": 2016,
    "Locobuzz": 2015,
    "Flipkart": 2007,
    "Mindtree": 1999
}

def validate_candidate_timeline(candidate):
    """
    Validates a candidate profile for chronological anomalies and honeypot indicators.
    Returns:
        (is_valid: bool, reason: str, anomalies: list)
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    edu = candidate.get("education", [])
    yoe = float(profile.get("years_of_experience", 0.0))
    
    anomalies = []
    current_date = datetime.date(2026, 6, 28) # Hackathon context date
    
    # 1. Startup founding date check & duration check
    for job in history:
        company = job.get("company")
        if company in STARTUP_FOUNDING:
            founded_year = STARTUP_FOUNDING[company]
            start_s = job.get("start_date")
            start_d = parse_date(start_s)
            if start_d and start_d.year < founded_year:
                anomalies.append(
                    f"Honeypot Indicator: Candidate worked at '{company}' starting in {start_s}, "
                    f"but the company was founded in {founded_year}."
                )
            
            # Check if duration exceeds company age
            duration_months = job.get("duration_months")
            if duration_months:
                max_possible_months = (current_date.year - founded_year) * 12 + 6
                if duration_months > max_possible_months:
                    anomalies.append(
                        f"Honeypot Indicator: Candidate worked at '{company}' for {duration_months} months, "
                        f"but the company was founded in {founded_year} (max possible duration: {max_possible_months} months)."
                    )
                
    # 2. Skill duration check
    fake_skills = []
    for skill in skills:
        prof = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)
        if prof in ["expert", "advanced"] and duration == 0:
            fake_skills.append(skill.get("name"))
            
    if len(fake_skills) >= 1:
        anomalies.append(
            f"Honeypot Indicator: Candidate lists expert/advanced skills "
            f"with 0 duration: {fake_skills}."
        )
        
    # 3. Basic timeline date overlap / consistency checks
    intervals = []
    start_dates = []
    for job in history:
        start_s = job.get("start_date")
        end_s = job.get("end_date")
        dur_reported = job.get("duration_months")
        company = job.get("company", "Unknown")
        
        start_d = parse_date(start_s)
        end_d = parse_date(end_s) if end_s else current_date
        
        if start_d:
            start_dates.append(start_d)
            
        if start_d and end_d:
            intervals.append((start_d, end_d, company))
            
            # Start date after end date
            if start_d > end_d:
                anomalies.append(f"Invalid Timeline: Job at '{company}' start date {start_s} is after end date {end_s}.")
                
            # Date in future
            if start_d > current_date:
                anomalies.append(f"Invalid Timeline: Job at '{company}' start date {start_s} is in the future.")
                
            # Duration mismatch
            delta_days = (end_d - start_d).days
            dur_calc = round(delta_days / 30.4375)
            if dur_reported is not None and abs(dur_calc - dur_reported) > 6:
                anomalies.append(
                    f"Timeline Inconsistency: Job at '{company}' duration reported as {dur_reported} months, "
                    f"but calculated duration is {dur_calc} months."
                )
                
    # 4. YoE exceeds elapsed time check
    if start_dates:
        first_start = min(start_dates)
        elapsed_years = (current_date - first_start).days / 365.25
        if yoe > elapsed_years + 0.5:
            anomalies.append(
                f"Honeypot Indicator: Candidate claims {yoe} YoE, but first job started on {first_start} "
                f"({elapsed_years:.2f} years ago)."
            )
            
    # 5. Senior roles before graduation check
    grad_years = [e.get("end_year") for e in edu if e.get("end_year")]
    min_grad_year = min(grad_years) if grad_years else None
    if min_grad_year:
        for job in history:
            start_s = job.get("start_date")
            start_d = parse_date(start_s)
            if start_d and start_d.year < min_grad_year - 2:
                title = job.get("title", "").lower()
                if any(x in title for x in ["senior", "lead", "manager", "architect", "director", "principal"]):
                    anomalies.append(
                        f"Chronological Contradiction: Held senior role '{job.get('title')}' in {start_d.year} "
                        f"but did not graduate college until {min_grad_year}."
                    )
                     
    # 6. Overlapping full-time jobs check
    overlaps = 0
    for i in range(len(intervals)):
        for j in range(i+1, len(intervals)):
            s1, e1, c1 = intervals[i]
            s2, e2, c2 = intervals[j]
            if max(s1, s2) < min(e1, e2):
                overlap_days = (min(e1, e2) - max(s1, s2)).days
                if overlap_days > 60:
                    overlaps += 1
                    
    if overlaps >= 2:
        anomalies.append(f"Inconsistent Profile: Multiple concurrent full-time job overlaps ({overlaps} instances).")

    is_valid = len(anomalies) == 0
    reason = "; ".join(anomalies) if anomalies else "Authentic profile"
    return is_valid, reason, anomalies

def calculate_resume_authenticity_score(candidate):
    """
    Evaluates profile authenticity. Combines timeline validations with platform verification signals.
    """
    is_valid, _, anomalies = validate_candidate_timeline(candidate)
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
        score -= 10.0
        
    # Account activity check
    signup_s = signals.get("signup_date")
    last_active_s = signals.get("last_active_date")
    signup_d = parse_date(signup_s)
    last_active_d = parse_date(last_active_s)
    
    if signup_d and last_active_d:
        if signup_d > last_active_d:
            score -= 20.0
            
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
                score -= 3.0
                
    return max(0.0, score)
