def combine_and_adjust_score(cid, lgb_score, sim_score, candidate_record, config=None):
    """
    Combines LightGBM score and FAISS similarity score.
    Applies location, notice period, title, and YoE adjustments.
    """
    profile = candidate_record.get("profile", {})
    current_title = profile.get("current_title", "").lower()
    signals = candidate_record.get("redrob_signals", {})
    
    # 1. Base combination (LGBM * 0.6 + FAISS * 0.4)
    final_score = lgb_score * 0.6 + sim_score * 0.4
    
    # 2. Location modifier
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    
    preferred_locs = ["pune", "noida", "delhi", "ncr", "gurgaon", "mumbai", "hyderabad"]
    is_preferred_loc = any(loc in location for loc in preferred_locs)
    if is_preferred_loc:
        final_score += 0.05
        
    if country != "india" and not willing_to_relocate:
        final_score -= 0.15
        
    # 3. Notice period modifier
    notice_period = int(signals.get("notice_period_days", 30))
    if notice_period <= 30:
        final_score += 0.05
    elif notice_period > 90:
        final_score -= 0.10
        
    # 4. Title modifier
    title_mod = 0.0
    ai_ml_keywords = [
        "ml", "machine learning", "ai ", "ai-", "artificial intelligence", 
        "data scientist", "nlp", "computer vision", "cv engineer", "deep learning", 
        "reinforcement learning", "recommendation", "search engineer", "retrieval",
        "applied scientist", "ai specialist", "ai research"
    ]
    if any(keyword in current_title or current_title.startswith("ai") or current_title.endswith("ai") for keyword in ai_ml_keywords):
        title_mod = 0.15
    elif any(k in current_title for k in ["backend", "data engineer", "analytics engineer"]):
        title_mod = 0.05
        
    final_score += title_mod
    
    # 5. Experience modifier
    yoe = float(profile.get("years_of_experience", 0.0))
    if 5.0 <= yoe <= 9.0:
        final_score += 0.05
    elif yoe < 3.0:
        final_score -= 0.15
    elif yoe > 15.0:
        final_score -= 0.10
        
    return final_score
