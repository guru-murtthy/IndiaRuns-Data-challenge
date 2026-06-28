#!/usr/bin/env python3
import os
import json
import argparse
import pickle
import pandas as pd
import numpy as np

from configs.app import get_config
from ingestion.candidate_loader import load_candidates
from ingestion.jd_loader import load_jd
from validation.timeline_validator import validate_candidate_timeline
from embeddings.encoder import CandidateEncoder
from embeddings.faiss_builder import load_faiss_index
from embeddings.retriever import retrieve_top_k
from ranking.scorer import predict_gbm_scores
from ranking.score_combiner import combine_and_adjust_score
from ranking.final_rank import rank_and_select_top_k
from reasoning.reasoning_engine import generate_reasoning
from submission.csv_writer import write_submission_csv
from utils.logger import log_info, log_error
from utils.timer import Timer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="data/raw/candidates.jsonl")
    parser.add_argument("--out", type=str, default="data/output/submission.csv")
    args = parser.parse_args()
    
    config = get_config()
    
    # Check candidates path and fallback
    if not os.path.exists(args.candidates):
        # Check standard fallback folder
        fallback_path = "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
        if os.path.exists(fallback_path):
            args.candidates = fallback_path
        elif os.path.exists("candidates.jsonl"):
            args.candidates = "candidates.jsonl"
            
    log_info("Loading ranking models and resources...")
    
    # 1. Load Embeddings Model
    encoder_path = config["embeddings"].get("local_model_path", "models/embeddings/sentence_transformer_model")
    if not os.path.exists(encoder_path) and os.path.exists("sentence_transformer_model"):
        encoder_path = "sentence_transformer_model"
        
    encoder = CandidateEncoder(local_dir=encoder_path)
    encoder.load_model()
    
    # 2. Load FAISS index
    faiss_path = config["embeddings"].get("faiss_index_path", "models/checkpoints/faiss_index.bin")
    if not os.path.exists(faiss_path) and os.path.exists("faiss_index.bin"):
        faiss_path = "faiss_index.bin"
        
    index = load_faiss_index(faiss_path)
    
    # 3. Load Candidate IDs
    cand_ids_path = config["embeddings"].get("candidate_ids_path", "models/checkpoints/candidate_ids.json")
    if not os.path.exists(cand_ids_path) and os.path.exists("candidate_ids.json"):
        cand_ids_path = "candidate_ids.json"
        
    with open(cand_ids_path, "r") as f:
        cand_ids = json.load(f)
        
    # 4. Load LightGBM model
    gbm_path = config["ranker"].get("model_path", "models/checkpoints/lightgbm_model.bin")
    if not os.path.exists(gbm_path) and os.path.exists("lightgbm_model.bin"):
        gbm_path = "lightgbm_model.bin"
        
    with open(gbm_path, "rb") as f:
        gbm = pickle.load(f)
        
    # 5. Load Candidate Features DataFrame
    features_path = config["ranker"].get("features_path", "data/processed/candidate_features.csv")
    if not os.path.exists(features_path) and os.path.exists("candidate_features.csv"):
        features_path = "candidate_features.csv"
        
    df_features = pd.read_csv(features_path)
    
    # Load target Job Description
    jd_query = load_jd()
    
    # 6. Embed Job Description
    log_info("Embedding Job Description...")
    jd_emb = encoder.encode([jd_query], show_progress_bar=False)
    jd_emb = np.array(jd_emb).astype('float32')
    
    # 7. Search FAISS Index
    log_info("Retrieving candidates using semantic search...")
    retrieved_cids, sim_dict = retrieve_top_k(index, jd_emb, k=1000, candidate_ids=cand_ids)
    
    # 8. Load full candidate records for validation
    log_info("Loading candidate records for validation...")
    try:
        candidates_list = load_candidates(args.candidates)
    except FileNotFoundError:
        log_error(f"Could not load candidates from {args.candidates}.")
        return
        
    candidate_records = {c.get("candidate_id"): c for c in candidates_list if c.get("candidate_id") in sim_dict}
    
    # 9. Filter and Score
    valid_candidates_scores = []
    feature_cols = [
        "years_of_experience", "skill_count", "skill_evidence_score", 
        "career_growth_score", "resume_authenticity_score", "profile_completeness_score", 
        "recruiter_response_rate", "avg_response_time_hours", "connection_count", 
        "endorsements_received", "notice_period_days", "salary_min", "salary_max", 
        "github_activity_score", "search_appearance_30d", "saved_by_recruiters_30d", 
        "interview_completion_rate", "offer_acceptance_rate", "is_consulting_only"
    ]
    
    skipped_honeypots = 0
    skipped_consulting = 0
    
    log_info("Running validation and model scoring...")
    
    # Predict GBM scores in batch for efficiency
    lgb_scores = predict_gbm_scores(gbm, df_features, feature_cols, retrieved_cids)
    
    blacklist = config["parser"].get("blacklist_titles", [])
    
    for cid in retrieved_cids:
        cand = candidate_records.get(cid)
        if not cand:
            continue
            
        # Strict Honeypot filter
        is_valid, _, _ = validate_candidate_timeline(cand)
        if not is_valid:
            skipped_honeypots += 1
            continue
            
        # Title relevance filter
        profile = cand.get("profile", {})
        current_title = profile.get("current_title", "").lower()
        if any(keyword in current_title for keyword in blacklist):
            continue
            
        lgb_score = lgb_scores.get(cid)
        if lgb_score is None:
            continue
            
        # consulting firm filtering
        row = df_features[df_features["candidate_id"] == cid]
        if not row.empty and row["is_consulting_only"].values[0] == 1.0:
            skipped_consulting += 1
            continue
            
        # Combine and adjust scores
        sim_score = sim_dict[cid]
        final_score = combine_and_adjust_score(cid, lgb_score, sim_score, cand)
        
        valid_candidates_scores.append({
            "candidate_id": cid,
            "score": final_score,
            "record": cand
        })
        
    log_info(f"Honeypots skipped during ranking: {skipped_honeypots}")
    log_info(f"Consulting-only candidates skipped: {skipped_consulting}")
    log_info(f"Total valid candidates for sorting: {len(valid_candidates_scores)}")
    
    # 10. Sort and select top 100
    top_100 = rank_and_select_top_k(valid_candidates_scores, k=100)
    
    # 11. Generate Reasonings
    final_candidates = []
    for rank_idx, item in enumerate(top_100):
        rank_num = rank_idx + 1
        cid = item["candidate_id"]
        score = item["score"]
        cand_record = item["record"]
        
        reasoning = generate_reasoning(cand_record, rank_num, score)
        final_candidates.append({
            "candidate_id": cid,
            "rank": rank_num,
            "score": score,
            "reasoning": reasoning
        })
        
    # Ensure directory exists for output
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    write_submission_csv(final_candidates, args.out)
    log_info(f"Successfully generated top 100 candidate ranking in {args.out}")

if __name__ == '__main__':
    main()
