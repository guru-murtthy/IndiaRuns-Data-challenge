#!/usr/bin/env python3
import json
import os
import argparse
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from utils import (
    validate_candidate_timeline,
    CONSULTING_FIRMS
)
from member3 import generate_reasoning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    parser.add_argument("--out", type=str, default="submission.csv")
    args = parser.parse_args()
    
    if not os.path.exists(args.candidates):
        args.candidates = "candidates.jsonl"
        
    print("Loading ranking models and indexes...")
    # Load model and resources offline
    model = SentenceTransformer("sentence_transformer_model")
    index = faiss.read_index("faiss_index.bin")
    with open("candidate_ids.json", "r") as f:
        cand_ids = json.load(f)
    with open("lightgbm_model.bin", "rb") as f:
        gbm = pickle.load(f)
    df_features = pd.read_csv("candidate_features.csv")
    
    # Target Job Description query representation
    jd_query = (
        "Senior ML Engineer Search Ranking Retrieval. "
        "Production experience with embeddings-based retrieval systems, sentence-transformers, vector search, FAISS, Pinecone, Qdrant, Milvus. "
        "Strong Python. Designing evaluation frameworks NDCG, MRR, MAP. 5-9 years experience, product companies. Pune, Noida."
    )
    
    # 1. Embed JD
    print("Embedding Job Description...")
    jd_emb = model.encode([jd_query], normalize_embeddings=True)
    jd_emb = np.array(jd_emb).astype('float32')
    
    # 2. Search FAISS Index
    # Search up to index size if index is smaller than 1000
    search_k = min(1000, index.ntotal)
    similarities, indices = index.search(jd_emb, search_k)
    
    retrieved_cids = []
    sim_dict = {}
    for idx, sim in zip(indices[0], similarities[0]):
        if idx == -1 or idx >= len(cand_ids):
            continue
        cid = cand_ids[idx]
        if cid not in sim_dict:
            sim_dict[cid] = sim
            retrieved_cids.append(cid)
    
    # 3. Load full candidate records for validation
    print("Loading candidate records for validation...")
    candidate_records = {}
    
    if args.candidates.endswith('.json'):
        with open(args.candidates, 'r', encoding='utf-8') as f:
            candidates_list = json.load(f)
            for cand in candidates_list:
                cid = cand.get("candidate_id")
                if cid in sim_dict:
                    candidate_records[cid] = cand
    else:
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split('"', 4)
                if len(parts) >= 4 and parts[1] == 'candidate_id':
                    cid = parts[3]
                    if cid in sim_dict:
                        candidate_records[cid] = json.loads(line)
                
    # 4. Filter and Score
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
    
    for cid in retrieved_cids:
        cand = candidate_records.get(cid)
        if not cand:
            continue
            
        # Strict Honeypot filter via Member 3 Validation Layer
        is_valid, _, _ = validate_candidate_timeline(cand)
        if not is_valid:
            skipped_honeypots += 1
            continue
            
        # Title relevance filter
        profile = cand.get("profile", {})
        current_title = profile.get("current_title", "").lower()
        
        # Blacklist of completely non-technical / irrelevant roles
        blacklist = [
            "graphic designer", "designer", "illustrator", "marketing", "hr ", "hr manager", "recruiter", 
            "human resources", "accountant", "accounting", "finance", "sales", "executive", 
            "customer support", "support agent", "helpdesk", "mechanical engineer", 
            "civil engineer", "operations manager", "project manager", "business analyst",
            "content writer", "writer"
        ]
        if any(keyword in current_title for keyword in blacklist):
            continue
            
        row = df_features[df_features["candidate_id"] == cid]
        if row.empty:
            continue
            
        # consulting firm filtering
        if row["is_consulting_only"].values[0] == 1.0:
            skipped_consulting += 1
            continue
            
        X_test = row[feature_cols]
        lgb_score = gbm.predict(X_test)[0]
        
        # Combine FAISS semantic similarity with LightGBM features
        sim_score = sim_dict[cid]
        final_score = lgb_score * 0.6 + sim_score * 0.4
        
        # Location modifier
        location = profile.get("location", "").lower()
        country = profile.get("country", "").lower()
        signals = cand.get("redrob_signals", {})
        willing_to_relocate = signals.get("willing_to_relocate", False)
        
        is_preferred_loc = any(loc in location for loc in ["pune", "noida", "delhi", "ncr", "gurgaon", "mumbai", "hyderabad"])
        if is_preferred_loc:
            final_score += 0.05
            
        if country != "india" and not willing_to_relocate:
            final_score -= 0.15
            
        # Notice period modifier
        notice_period = int(signals.get("notice_period_days", 30))
        if notice_period <= 30:
            final_score += 0.05
        elif notice_period > 90:
            final_score -= 0.10
            
        # Title modifier (AI/ML roles get a significant boost, backend/data secondary boost)
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
        
        # Experience modifier (5-9 years YoE is preferred, penalize too low or extremely high)
        yoe = float(profile.get("years_of_experience", 0.0))
        if 5.0 <= yoe <= 9.0:
            final_score += 0.05
        elif yoe < 3.0:
            final_score -= 0.15
        elif yoe > 15.0:
            final_score -= 0.10
            
        valid_candidates_scores.append({
            "candidate_id": cid,
            "score": final_score,
            "record": cand
        })
        
    print(f"Honeypots skipped during ranking: {skipped_honeypots}")
    print(f"Consulting-only candidates skipped: {skipped_consulting}")
    print(f"Total valid candidates: {len(valid_candidates_scores)}")
    
    # 5. Deterministic sorting: rounded score desc, candidate_id asc to satisfy tie-breaker
    valid_candidates_scores.sort(key=lambda x: (-round(x["score"], 6), x["candidate_id"]))
    
    # Top 100
    top_100 = valid_candidates_scores[:100]
    
    # 6. Generate CSV Rows
    csv_rows = []
    for rank_idx, item in enumerate(top_100):
        rank = rank_idx + 1
        cid = item["candidate_id"]
        score = round(item["score"], 6)
        cand_record = item["record"]
        
        reasoning = generate_reasoning(cand_record, rank, score)
        csv_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": score,
            "reasoning": reasoning
        })
        
    df_out = pd.DataFrame(csv_rows)
    df_out = df_out[["candidate_id", "rank", "score", "reasoning"]]
    
    df_out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Successfully generated top 100 candidate ranking in {args.out}")

if __name__ == '__main__':
    main()
