import pytest
from ranking.feature_engineering import FeatureEngineer

class TestFeatureEngineering:

    @pytest.fixture
    def engineer(self):
        return FeatureEngineer()

    def test_build_features(self, engineer):
        jd = {
            "title": "Senior Python Engineer",
            "experience": "5 years",
            "required_skills": ["Python", "FastAPI"],
            "preferred_skills": ["Docker", "AWS"],
            "location": "Bangalore"
        }
        
        cand = {
            "title": "Python Developer",
            "experience": "6 yrs",
            "skills": ["Python", "Django", "FastAPI", "Docker"],
            "company": "Tech Corp",
            "education": "B.Tech",
            "location": "Bangalore, India",
            "certifications": ["AWS Solutions Architect"]
        }
        
        fv = engineer.build_features(jd, cand, retrieval_score=0.9)
        features = fv.to_dict()
        
        assert features["semantic_similarity"] == 0.9
        assert features["required_skill_overlap"] == 1.0 # 2/2
        assert features["preferred_skill_overlap"] == 0.5 # 1/2
        assert features["missing_required_skills"] == 0.0
        assert features["experience_difference"] == 1.0 # 6 - 5
        assert features["relevant_experience"] == 1.0 # 6/5 capped at 1.0
        assert features["company_score"] == 0.8
        assert features["education_score"] == 0.6
        assert features["location_match"] == 1.0
        assert features["certification_score"] == 1.0
