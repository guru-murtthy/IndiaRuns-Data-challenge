def rank_and_select_top_k(candidates_scores, k=100):
    """
    Sorts candidate list deterministically:
    1. Rounded final score (to 6 decimal places) in descending order.
    2. Candidate ID in ascending order (tie-breaker).
    Returns top K candidates.
    """
    # Deterministic sorting
    candidates_scores.sort(key=lambda x: (-round(x["score"], 6), x["candidate_id"]))
    return candidates_scores[:k]
