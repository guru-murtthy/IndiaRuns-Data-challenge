import pandas as pd

def predict_gbm_scores(gbm, df_features, feature_cols, retrieved_cids):
    """
    Looks up precomputed feature vectors and runs LightGBM predictions for retrieved candidates.
    Returns a dictionary of candidate_id -> lgb_score.
    """
    # Filter features for retrieved candidates
    df_cand_feats = df_features[df_features["candidate_id"].isin(retrieved_cids)].copy()
    if df_cand_feats.empty:
        return {}
        
    X = df_cand_feats[feature_cols]
    preds = gbm.predict(X)
    
    lgb_scores = {}
    for cid, score in zip(df_cand_feats["candidate_id"], preds):
        lgb_scores[cid] = float(score)
        
    return lgb_scores
