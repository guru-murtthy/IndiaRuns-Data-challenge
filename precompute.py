import json
import os
import argparse
import pandas as pd
import numpy as np
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
from utils import (
    validate_candidate_timeline,
    calculate_skill_evidence_score,
    calculate_career_growth_score,
    calculate_resume_authenticity_score,
    CONSULTING_FIRMS
)

# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------

def extract_features(cand):
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    # Check if candidate has only worked at consulting/service firms
    companies = [job.get("company", "") for job in history if job.get("company")]
    is_consulting_only = 1.0 if companies and all(c in CONSULTING_FIRMS for c in companies) else 0.0
    
    # Expected salary
    expected_salary = signals.get("expected_salary_range_inr_lpa", {})
    salary_min = expected_salary.get("min", 15.0)
    salary_max = expected_salary.get("max", 30.0)
    
    # Calculate custom scores using Member 3 package
    skill_evidence = calculate_skill_evidence_score(cand)
    career_growth = calculate_career_growth_score(cand)
    authenticity = calculate_resume_authenticity_score(cand)
    
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

# ---------------------------------------------------------
# Main Precomputation Pipeline
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="candidates.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    if not os.path.exists(args.candidates):
        args.candidates = "candidates.jsonl"
        
    print(f"Reading candidates from {args.candidates}...")
    
    if args.candidates.endswith('.json'):
        with open(args.candidates, 'r', encoding='utf-8') as f:
            candidates_data = json.load(f)
    else:
        candidates_data = []
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates_data.append(json.loads(line))
                    
    valid_candidates = []
    skipped_honeypots = 0
    
    for idx, cand in enumerate(candidates_data):
        if args.limit and idx >= args.limit:
            break
            
        # Filter out honeypots using Member 3 Validation Layer
        is_valid, _, _ = validate_candidate_timeline(cand)
        if not is_valid:
            skipped_honeypots += 1
            continue
            
        valid_candidates.append(cand)
        
        if len(valid_candidates) % 10000 == 0:
            print(f"Loaded {len(valid_candidates)} valid candidates...")
                
    print(f"Loaded total {len(valid_candidates)} valid candidates (skipped {skipped_honeypots} honeypots).")
    
    # 1. Feature Engineering
    print("Extracting features...")
    feature_list = []
    text_representations = []
    
    for cand in valid_candidates:
        feats = extract_features(cand)
        feature_list.append(feats)
        
        # Text for sentence embedding
        profile = cand.get("profile", {})
        skills_text = ", ".join([s.get("name", "") for s in cand.get("skills", [])])[:120]
        text_rep = f"{profile.get('current_title', '')} | {profile.get('headline', '')}. Skills: {skills_text}"
        text_representations.append(text_rep)
        
    df_features = pd.DataFrame(feature_list)
    print(f"Features DataFrame shape: {df_features.shape}")
    
    # 2. Embedding Generation
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Saving SentenceTransformer model locally...")
    model.save("sentence_transformer_model")
    
    print("Generating embeddings (this might take a few minutes on CPU)...")
    embeddings = model.encode(text_representations, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    print(f"Generated embeddings of shape {embeddings.shape}")
    
    # 3. Build FAISS Index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, "faiss_index.bin")
    cand_ids = df_features["candidate_id"].tolist()
    with open("candidate_ids.json", "w") as f:
        json.dump(cand_ids, f)
        
    print("FAISS index and ID mapping saved.")
    
    # 4. Train Self-Supervised LightGBM Ranker
    print("Training self-supervised LightGBM model...")
    import lightgbm as lgb
    
    target_score = (
        df_features["skill_evidence_score"] * 0.25 +
        df_features["career_growth_score"] * 0.15 +
        (df_features["resume_authenticity_score"] / 100.0) * 0.20 +
        (1.0 - df_features["notice_period_days"] / 180.0) * 0.10 +
        df_features["recruiter_response_rate"] * 0.15 +
        (np.clip(df_features["github_activity_score"], 0, 100) / 100.0) * 0.05 +
        (df_features["profile_completeness_score"] / 100.0) * 0.10
    )
    target_score = (target_score - target_score.min()) / (target_score.max() - target_score.min() + 1e-9)
    
    feature_cols = [
        "years_of_experience", "skill_count", "skill_evidence_score", 
        "career_growth_score", "resume_authenticity_score", "profile_completeness_score", 
        "recruiter_response_rate", "avg_response_time_hours", "connection_count", 
        "endorsements_received", "notice_period_days", "salary_min", "salary_max", 
        "github_activity_score", "search_appearance_30d", "saved_by_recruiters_30d", 
        "interview_completion_rate", "offer_acceptance_rate", "is_consulting_only"
    ]
    
    X = df_features[feature_cols]
    y = target_score
    
    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "verbosity": -1,
        "n_jobs": 4
    }
    
    gbm = lgb.train(params, train_data, num_boost_round=100)
    
    with open("lightgbm_model.bin", "wb") as f:
        pickle.dump(gbm, f)
        
    df_features.to_csv("candidate_features.csv", index=False)
    print("Precomputation completed successfully.")

if __name__ == '__main__':
    main()
