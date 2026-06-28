def validate_candidate_schema(candidate):
    """
    Checks if a candidate dictionary has the required top-level keys.
    """
    required_keys = ["candidate_id", "profile", "career_history", "education", "skills", "redrob_signals"]
    for key in required_keys:
        if key not in candidate:
            return False, f"Missing required key: {key}"
    return True, ""
