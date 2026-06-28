import pandas as pd

def write_submission_csv(top_candidates, out_path):
    """
    Writes the final top candidates list to a UTF-8 encoded CSV.
    Headers must be exactly: candidate_id, rank, score, reasoning
    """
    csv_rows = []
    for item in top_candidates:
        csv_rows.append({
            "candidate_id": item["candidate_id"],
            "rank": item["rank"],
            "score": round(item["score"], 6),
            "reasoning": item["reasoning"]
        })
        
    df_out = pd.DataFrame(csv_rows)
    df_out = df_out[["candidate_id", "rank", "score", "reasoning"]]
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    return len(df_out)
