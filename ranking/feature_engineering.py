import logging
from typing import Dict, Any, List
import re

from ranking.feature_vector import FeatureVector

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Computes numerical features for ranking candidates against a JD.
    """
    
    def __init__(self):
        pass

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _extract_years(self, text: str) -> float:
        """
        Naive extraction of years from a string.
        Returns 0.0 if not found.
        """
        if not text:
            return 0.0
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years|yrs|y)', str(text).lower())
        if match:
            return self._safe_float(match.group(1))
        # fallback for just a number
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
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def build_features(
        self, 
        jd: Dict[str, Any], 
        candidate: Dict[str, Any],
        retrieval_score: float = 0.0
    ) -> FeatureVector:
        """
        Builds the feature vector for a given candidate and JD.
        
        Args:
            jd: The Job Description data dictionary.
            candidate: The Candidate data dictionary.
            retrieval_score: The similarity score from the retrieval stage (if available).
            
        Returns:
            A FeatureVector object containing all numerical features.
        """
        features = {}

        # 1. Semantic similarity (passed from retrieval)
        features["semantic_similarity"] = retrieval_score

        # Ensure we have skills
        jd_skills_req = set(s.lower() for s in jd.get("required_skills", []))
        jd_skills_pref = set(s.lower() for s in jd.get("preferred_skills", []))
        cand_skills = set(s.lower() for s in candidate.get("skills", []))
        
        # 2. Required skill overlap %
        features["required_skill_overlap"] = self._calculate_overlap(jd_skills_req, cand_skills)
        
        # 3. Preferred skill overlap %
        features["preferred_skill_overlap"] = self._calculate_overlap(jd_skills_pref, cand_skills)

        # 4. Missing required skills (count)
        missing_skills = len(jd_skills_req) - len(jd_skills_req.intersection(cand_skills))
        features["missing_required_skills"] = float(missing_skills)

        # Experience
        jd_exp_str = jd.get("experience", "0")
        cand_exp_str = candidate.get("experience", "0")
        
        jd_exp = self._extract_years(jd_exp_str)
        cand_exp = self._extract_years(cand_exp_str)
        
        # 5. Experience difference
        features["experience_difference"] = cand_exp - jd_exp
        
        # 6. Relevant experience score (heuristic: cap diff)
        # If candidate has more or equal exp, score is 1.0. If less, penalize linearly.
        if jd_exp > 0:
            features["relevant_experience"] = max(0.0, min(1.0, cand_exp / jd_exp))
        else:
            features["relevant_experience"] = 1.0 if cand_exp > 0 else 0.0

        # 7. Current title similarity (Jaccard)
        jd_title = str(jd.get("title", ""))
        cand_title = str(candidate.get("title", ""))
        features["title_similarity"] = self._jaccard_similarity(jd_title, cand_title)

        # 8. Company quality score (dummy heuristic: 0.5 if company present)
        cand_company = candidate.get("company")
        features["company_score"] = 0.8 if cand_company else 0.2

        # 9. Education score (dummy heuristic)
        cand_edu = str(candidate.get("education", "")).lower()
        if "b.tech" in cand_edu or "b.e" in cand_edu or "bachelor" in cand_edu:
            features["education_score"] = 0.6
        elif "m.tech" in cand_edu or "master" in cand_edu or "ms" in cand_edu:
            features["education_score"] = 0.8
        elif "phd" in cand_edu:
            features["education_score"] = 1.0
        else:
            features["education_score"] = 0.3 if cand_edu else 0.0

        # 10. Location match
        jd_loc = str(jd.get("location", "")).lower()
        cand_loc = str(candidate.get("location", "")).lower()
        features["location_match"] = 1.0 if (jd_loc and jd_loc in cand_loc) else (0.5 if cand_loc else 0.0)
        
        # 11. Certification score
        features["certification_score"] = 1.0 if candidate.get("certifications") else 0.0
        
        # 12. Project relevance score (dummy overlap based)
        jd_desc = str(jd.get("description", "")).lower()
        cand_proj = str(candidate.get("projects", "")).lower()
        features["project_relevance"] = min(1.0, len(set(cand_proj.split()).intersection(set(jd_desc.split()))) / 10.0)

        # 13. Notice period score (lower is better, assuming < 30 days is optimal)
        np_str = candidate.get("notice_period", "90")
        notice_days = self._extract_years(np_str) # extract numbers
        if notice_days <= 0:
            notice_days = 90.0 # Default fallback
            
        if notice_days <= 15:
            features["notice_period_score"] = 1.0
        elif notice_days <= 30:
            features["notice_period_score"] = 0.8
        elif notice_days <= 60:
            features["notice_period_score"] = 0.5
        else:
            features["notice_period_score"] = 0.2

        return FeatureVector(features=features)
