import logging
from typing import List, Dict, Any, Union

from ranking.feature_engineering import FeatureEngineer
from ranking.lightgbm_ranker import LightGBMRanker
from ranking.catboost_ranker import CatBoostRanker
from ranking.scorer import RankingScorer
from ranking.calibration import ScoreCalibration

logger = logging.getLogger(__name__)

class Reranker:
    """
    Coordinates the reranking pipeline: feature generation, prediction, calibration, and sorting.
    """
    
    def __init__(self, model_type: str = "lightgbm", model_path: str = None, calibration_method: str = "min_max"):
        """
        Initializes the Reranker.
        
        Args:
            model_type: 'lightgbm', 'catboost', or 'fallback'
            model_path: Path to the trained model.
            calibration_method: 'min_max', 'z_score', 'temperature', or None
        """
        self.feature_engineer = FeatureEngineer()
        self.calibration_method = calibration_method
        
        if model_type == "lightgbm":
            self.ranker = LightGBMRanker(model_path)
        elif model_type == "catboost":
            self.ranker = CatBoostRanker(model_path)
        else:
            self.ranker = RankingScorer()
            
        logger.info(f"Initialized Reranker with model: {model_type}, calibration: {calibration_method}")

    def rerank(self, jd: Dict[str, Any], top_k_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved Top-K candidates.
        
        Args:
            jd: The Job Description dictionary.
            top_k_candidates: List of candidate dictionaries returned by retrieval.
            
        Returns:
            List of candidate dictionaries augmented with 'rerank_score', sorted descending.
        """
        if not top_k_candidates:
            return []

        logger.info(f"Reranking {len(top_k_candidates)} candidates.")
        
        # 1. Feature Generation
        feature_vectors = []
        for cand in top_k_candidates:
            retrieval_score = float(cand.get("retrieval_score", 0.0))
            fv = self.feature_engineer.build_features(jd, cand, retrieval_score=retrieval_score)
            feature_vectors.append(fv)

        # 2. Predict Scores
        if hasattr(self.ranker, 'predict_batch'):
            raw_scores = self.ranker.predict_batch(feature_vectors)
        else:
            raw_scores = [self.ranker.score(fv) for fv in feature_vectors]

        # 3. Calibration
        calibrated_scores = raw_scores
        if self.calibration_method == "min_max":
            calibrated_scores = ScoreCalibration.min_max_normalize(raw_scores)
        elif self.calibration_method == "z_score":
            calibrated_scores = ScoreCalibration.z_score_normalize(raw_scores)
        elif self.calibration_method == "temperature":
            calibrated_scores = ScoreCalibration.temperature_scaling(raw_scores)
            
        # 4. Augment and Sort
        reranked_candidates = []
        for cand, score in zip(top_k_candidates, calibrated_scores):
            # Create a shallow copy to avoid mutating the original input if needed
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = float(score)
            reranked_candidates.append(cand_copy)
            
        # Sort descending by rerank_score
        reranked_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info("Reranking completed.")
        return reranked_candidates
