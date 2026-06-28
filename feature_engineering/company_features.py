CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini", "HCL", "Tech Mahindra", "Mphasis"
}

def is_consulting_only_candidate(candidate):
    """
    Checks if a candidate has only worked at consulting/service firms.
    """
    history = candidate.get("career_history", [])
    companies = [job.get("company", "") for job in history if job.get("company")]
    if companies and all(c in CONSULTING_FIRMS for c in companies):
        return 1.0
    return 0.0
