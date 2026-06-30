import logging
from typing import Dict, Any, List
import re

from feature_engineering.feature_vector import FeatureVector

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _extract_years(self, text: str) -> float:
        if not text:
            return 0.0
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years|yrs|y)', str(text).lower())
        if match:
            return self._safe_float(match.group(1))
        match = re.search(r'^(\d+(?:\.\d+)?)$', str(text).strip())
        if match:
             return self._safe_float(match.group(1))
        return 0.0

    def _calculate_overlap(self, set1: set, set2: set) -> float:
        if not set1:
            return 0.0
        return len(set1.intersection(set2)) / len(set1)

    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        if not str1 or not str2:
            return 0.0
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def build_features(self, jd: Dict[str, Any], candidate: Dict[str, Any], retrieval_score: float = 0.0) -> FeatureVector:
        features = {}
        features["semantic_similarity"] = retrieval_score

        jd_skills_req = set(s.lower() for s in jd.get("required_skills", []))
        cand_skills = set(s.lower() for s in candidate.get("skills", []))
        
        features["required_skill_overlap"] = self._calculate_overlap(jd_skills_req, cand_skills)
        features["missing_required_skills"] = float(len(jd_skills_req) - len(jd_skills_req.intersection(cand_skills)))

        jd_exp = self._safe_float(jd.get("min_experience_years", 0))
        cand_exp = self._safe_float(candidate.get("experience_years", 0))
        features["experience_difference"] = cand_exp - jd_exp
        
        if jd_exp > 0:
            features["relevant_experience"] = max(0.0, min(1.0, cand_exp / jd_exp))
        else:
            features["relevant_experience"] = 1.0 if cand_exp > 0 else 0.0

        jd_title = str(jd.get("title", ""))
        cand_title = str(candidate.get("title", ""))
        features["title_similarity"] = self._jaccard_similarity(jd_title, cand_title)
        
        cand_loc = str(candidate.get("location", "")).lower()
        features["location_match"] = 1.0 if cand_loc else 0.0

        return FeatureVector(features=features)
