import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from member3 import (
    validate_candidate_timeline,
    calculate_skill_evidence_score,
    calculate_career_growth_score,
    calculate_resume_authenticity_score,
    generate_reasoning
)

def run_analysis(candidates_path, limit=10000):
    print(f"=== Starting Member 3 Dataset Analysis (first {limit} candidates) ===")
    print(f"Reading dataset from: {candidates_path}")
    
    total_scanned = 0
    honeypot_anomalies_count = 0
    anomaly_types = Counter()
    
    skill_evidence_scores = []
    career_growth_scores = []
    authenticity_scores = []
    
    sample_honeypots = []
    sample_high = []
    sample_mid = []
    sample_low = []
    
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if total_scanned >= limit:
                break
            cand = json.loads(line)
            total_scanned += 1
            
            # 1. Run Timeline Validation
            is_valid, reason, anomalies = validate_candidate_timeline(cand)
            
            # Calculate scoring features
            evidence_s = calculate_skill_evidence_score(cand)
            growth_s = calculate_career_growth_score(cand)
            auth_s = calculate_resume_authenticity_score(cand)
            
            skill_evidence_scores.append(evidence_s)
            career_growth_scores.append(growth_s)
            authenticity_scores.append(auth_s)
            
            if not is_valid:
                honeypot_anomalies_count += 1
                for anomaly in anomalies:
                    # extract simplified type
                    if "Honeypot Indicator" in anomaly:
                        if "worked at startup" in anomaly.lower() or "started in" in anomaly.lower():
                            anomaly_types["Startup founding date contradiction"] += 1
                        else:
                            anomaly_types["Expert skill with 0 duration"] += 1
                    elif "Chronological Contradiction" in anomaly:
                        anomaly_types["Senior role before graduation"] += 1
                    elif "Invalid Timeline" in anomaly:
                        anomaly_types["Job start date after end date / future date"] += 1
                    elif "Timeline Inconsistency" in anomaly:
                        anomaly_types["Job duration mismatch"] += 1
                    elif "Inconsistent Profile" in anomaly:
                        anomaly_types["Overlapping job periods"] += 1
                    else:
                        anomaly_types["Other timeline issues"] += 1
                        
                if len(sample_honeypots) < 3:
                    sample_honeypots.append((cand, reason))
            else:
                # Classify by authenticity and growth scores for sample output
                # High: Authenticity > 90, Growth > 4
                if auth_s >= 90 and growth_s >= 4.0 and len(sample_high) < 3:
                    sample_high.append((cand, evidence_s, growth_s, auth_s))
                # Mid: Authenticity > 80, Growth 1-4
                elif auth_s >= 80 and 1.5 <= growth_s < 4.0 and len(sample_mid) < 3:
                    sample_mid.append((cand, evidence_s, growth_s, auth_s))
                # Low: Authenticity < 70 or Growth < 1.5
                elif (auth_s < 70 or growth_s < 1.5) and len(sample_low) < 3:
                    sample_low.append((cand, evidence_s, growth_s, auth_s))
                    
    # Generate statistics summary
    print("\n--- Validation Statistics ---")
    print(f"Total Candidates Scanned: {total_scanned}")
    print(f"Honeypots/Anomalous Profiles Flagged: {honeypot_anomalies_count} ({honeypot_anomalies_count/total_scanned*100:.2f}%)")
    print("\nBreakdown of Detected Anomalies:")
    for a_type, count in anomaly_types.items():
        print(f"  - {a_type}: {count}")
        
    print("\n--- Scoring Features Distribution ---")
    print(f"Skill Evidence Score:  Mean={np.mean(skill_evidence_scores):.3f}, Min={np.min(skill_evidence_scores):.3f}, Max={np.max(skill_evidence_scores):.3f}")
    print(f"Career Growth Score:   Mean={np.mean(career_growth_scores):.3f}, Min={np.min(career_growth_scores):.3f}, Max={np.max(career_growth_scores):.3f}")
    print(f"Authenticity Score:    Mean={np.mean(authenticity_scores):.3f}, Min={np.min(authenticity_scores):.3f}, Max={np.max(authenticity_scores):.3f}")
    
    # Print examples of Honeypots
    print("\n--- Sample Flagged Honeypots ---")
    for idx, (cand, reason) in enumerate(sample_honeypots):
        print(f"\nHoneypot Example {idx+1}: {cand.get('candidate_id')} ({cand['profile'].get('anonymized_name')})")
        print(f"  Current Title: {cand['profile'].get('current_title')}")
        print(f"  Validation Flags: {reason}")
        
    # Print examples of High Scoring Candidates
    print("\n--- Sample High-Fit Candidates & Generated Reasonings ---")
    for idx, (cand, ev, gr, au) in enumerate(sample_high):
        # Let's say this candidate is rank 1-3
        reasoning = generate_reasoning(cand, idx + 1, 0.95)
        print(f"\nCandidate {idx+1}: {cand.get('candidate_id')} ({cand['profile'].get('anonymized_name')})")
        print(f"  Title: {cand['profile'].get('current_title')} at {cand['profile'].get('current_company')}")
        print(f"  YoE: {cand['profile'].get('years_of_experience')}")
        print(f"  Scores: Skill Evidence={ev:.2f}, Career Growth={gr:.2f}, Authenticity={au:.2f}")
        print(f"  Generated Reasoning (Rank {idx+1}): \"{reasoning}\"")
        
    # Print examples of Mid-Fit Candidates
    print("\n--- Sample Mid-Fit Candidates & Generated Reasonings ---")
    for idx, (cand, ev, gr, au) in enumerate(sample_mid):
        reasoning = generate_reasoning(cand, idx + 20, 0.75)
        print(f"\nCandidate {idx+1}: {cand.get('candidate_id')} ({cand['profile'].get('anonymized_name')})")
        print(f"  Title: {cand['profile'].get('current_title')} at {cand['profile'].get('current_company')}")
        print(f"  Scores: Skill Evidence={ev:.2f}, Career Growth={gr:.2f}, Authenticity={au:.2f}")
        print(f"  Generated Reasoning (Rank {idx+20}): \"{reasoning}\"")
        
    # Print examples of Low-Fit Candidates
    print("\n--- Sample Low-Fit Candidates & Generated Reasonings ---")
    for idx, (cand, ev, gr, au) in enumerate(sample_low):
        reasoning = generate_reasoning(cand, idx + 80, 0.45)
        print(f"\nCandidate {idx+1}: {cand.get('candidate_id')} ({cand['profile'].get('anonymized_name')})")
        print(f"  Title: {cand['profile'].get('current_title')} at {cand['profile'].get('current_company')}")
        print(f"  Scores: Skill Evidence={ev:.2f}, Career Growth={gr:.2f}, Authenticity={au:.2f}")
        print(f"  Generated Reasoning (Rank {idx+80}): \"{reasoning}\"")

if __name__ == '__main__':
    # Determine the dataset path
    candidates_path = "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
    if not os.path.exists(candidates_path):
        candidates_path = "candidates.jsonl"
        
    if os.path.exists(candidates_path):
        run_analysis(candidates_path, 100000)
    else:
        print("Dataset not found at the expected paths. Make sure you extracted it.")
