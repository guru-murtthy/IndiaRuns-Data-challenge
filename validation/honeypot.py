from .timeline_validator import validate_candidate_timeline

def is_honeypot(candidate):
    """
    Checks if a candidate profile is classified as a honeypot (fails timeline checks).
    """
    is_valid, reason, _ = validate_candidate_timeline(candidate)
    return not is_valid, reason
