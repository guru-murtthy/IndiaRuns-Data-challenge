#!/usr/bin/env python3
import os
import json
import argparse
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb

from configs.app import get_config
from ingestion.candidate_loader import load_candidates
from validation.timeline_validator import validate_candidate_timeline, calculate_resume_authenticity_score
from feature_engineering.builder import extract_features
from embeddings.encoder import CandidateEncoder
from embeddings.faiss_builder import build_flat_ip_index, save_faiss_index
from utils.logger import log_info, log_error
from utils.timer import Timer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="data/raw/candidates.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    config = get_config()
    
    # 1. Load candidates
    log_info(f"Loading candidates from {args.candidates}...")
    try:
        candidates = load_candidates(args.candidates)
    except FileNotFoundError:
        # Fallback to check nested folder
        fallback_path = "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
        if os.path.exists(fallback_path):
            log_info(f"Fallback to loading candidates from {fallback_path}...")
            candidates = load_candidates(fallback_path)
        else:
            log_error(f"Could not find candidates dataset in data/raw/ or root/fallback.")
            return

    # Create directories if they don't exist
    os.makedirs("models/embeddings", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    valid_candidates = []
    skipped_honeypots = 0
    
    with Timer("Timeline Validation & Honeypot Filtering"):
        for idx, cand in enumerate(candidates):
            if args.limit and idx >= args.limit:
                break
            is_valid, _, _ = validate_candidate_timeline(cand)
            if not is_valid:
                skipped_honeypots += 1
                continue
            valid_candidates.append(cand)
            
    log_info(f"Loaded total {len(valid_candidates)} valid candidates (skipped {skipped_honeypots} honeypots).")
    
    # 2. Extract Features
    log_info("Extracting features...")
    feature_list = []
    text_representations = []
    
    with Timer("Feature Extraction"):
        for cand in valid_candidates:
            feats = extract_features(cand, calculate_resume_authenticity_score)
            feature_list.append(feats)
            
            profile = cand.get("profile", {})
            skills_text = ", ".join([s.get("name", "") for s in cand.get("skills", [])])[:120]
            text_rep = f"{profile.get('current_title', '')} | {profile.get('headline', '')}. Skills: {skills_text}"
            text_representations.append(text_rep)
            
    df_features = pd.DataFrame(feature_list)
    log_info(f"Features DataFrame shape: {df_features.shape}")
    
    # 3. Generate Embeddings & Save local ST model
    log_info("Initializing embedding encoder...")
    encoder = CandidateEncoder(
        model_name=config["embeddings"].get("model_name", "all-MiniLM-L6-v2"),
        local_dir=config["embeddings"].get("local_model_path", "models/embeddings/sentence_transformer_model")
    )
    
    with Timer("Embedding Generation & ST Save"):
        encoder.save_model() # Save model locally for offline execution
        embeddings = encoder.encode(text_representations, batch_size=config["embeddings"].get("batch_size", 256), show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
    log_info(f"Generated embeddings of shape {embeddings.shape}")
    
    # 4. Build FAISS Index
    log_info("Building and saving FAISS index...")
    with Timer("FAISS Builder"):
        index = build_flat_ip_index(embeddings)
        save_faiss_index(index, config["embeddings"].get("faiss_index_path", "models/checkpoints/faiss_index.bin"))
        
        cand_ids = df_features["candidate_id"].tolist()
        with open(config["embeddings"].get("candidate_ids_path", "models/checkpoints/candidate_ids.json"), "w") as f:
            json.dump(cand_ids, f)
            
    log_info("FAISS index and ID mapping saved successfully.")
    
    # 5. Train LightGBM model
    log_info("Training self-supervised LightGBM model...")
    with Timer("LightGBM Training"):
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
            "learning_rate": config["ranker"].get("learning_rate", 0.1),
            "num_leaves": config["ranker"].get("num_leaves", 31),
            "max_depth": config["ranker"].get("max_depth", -1),
            "verbosity": -1,
            "n_jobs": 4
        }
        
        gbm = lgb.train(params, train_data, num_boost_round=config["ranker"].get("num_boost_round", 100))
        
        with open(config["ranker"].get("model_path", "models/checkpoints/lightgbm_model.bin"), "wb") as f:
            pickle.dump(gbm, f)
            
        df_features.to_csv(config["ranker"].get("features_path", "data/processed/candidate_features.csv"), index=False)
        
    log_info("Precomputation and training completed successfully.")

if __name__ == '__main__':
    main()
