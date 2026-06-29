from typing import Dict, Optional
import logging
from ranking.feature_vector import FeatureVector

logger = logging.getLogger(__name__)

class RankingScorer:
    """
    Fallback linear weighting scorer when ML models (LightGBM/CatBoost) are not available.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initializes the RankingScorer with optional custom weights.
        """
        self.weights = weights or {
            "semantic_similarity": 0.4,
            "required_skill_overlap": 0.2,
            "preferred_skill_overlap": 0.1,
            "missing_required_skills": -0.1,
            "experience_difference": 0.05,
            "relevant_experience": 0.1,
            "title_similarity": 0.1,
            "company_score": 0.05,
            "education_score": 0.05,
            "location_match": 0.05,
            "certification_score": 0.02,
            "project_relevance": 0.08,
            "notice_period_score": 0.1
        }
        logger.info(f"Initialized RankingScorer with weights: {self.weights}")

    def score(self, feature_vector: FeatureVector) -> float:
        """
        Calculates a weighted score for a given feature vector.
        
        Args:
            feature_vector: The FeatureVector instance.
            
        Returns:
            A raw float score.
        """
        features_dict = feature_vector.to_dict()
        total_score = 0.0
        
        for feature_name, value in features_dict.items():
            weight = self.weights.get(feature_name, 0.0)
            total_score += float(value) * weight
            
        return total_score
