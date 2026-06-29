import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from ranking.feature_vector import FeatureVector
from ranking.scorer import RankingScorer

logger = logging.getLogger(__name__)

class LightGBMRanker:
    """
    Wrapper for LightGBM model to predict ranking scores.
    Falls back to RankingScorer if LightGBM is not installed or model is unavailable.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.fallback_scorer = RankingScorer()
        
        if lgb is None:
            logger.warning("LightGBM is not installed. Will use fallback scorer.")
        elif model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning(f"LightGBM model not found at {model_path}. Will use fallback scorer.")

    def load_model(self, path: str) -> None:
        """Loads a trained LightGBM model from disk."""
        if lgb is not None:
            try:
                self.model = lgb.Booster(model_file=path)
                logger.info(f"Successfully loaded LightGBM model from {path}")
            except Exception as e:
                logger.error(f"Failed to load LightGBM model from {path}: {e}")
                self.model = None
                
    def save_model(self, path: str) -> None:
        """Saves the trained LightGBM model to disk."""
        if self.model is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(path)
            logger.info(f"Successfully saved LightGBM model to {path}")
        else:
            logger.warning("No model to save.")

    def predict(self, feature_vector: FeatureVector) -> float:
        """Predicts a score for a single feature vector."""
        if self.model is not None:
            arr = feature_vector.to_numpy().reshape(1, -1)
            score = self.model.predict(arr)[0]
            return float(score)
        else:
            return self.fallback_scorer.score(feature_vector)

    def predict_batch(self, feature_vectors: List[FeatureVector]) -> List[float]:
        """Predicts scores for a batch of feature vectors."""
        if not feature_vectors:
            return []
            
        if self.model is not None:
            arr = np.array([fv.to_numpy() for fv in feature_vectors])
            scores = self.model.predict(arr)
            return [float(s) for s in scores]
        else:
            return [self.fallback_scorer.score(fv) for fv in feature_vectors]
