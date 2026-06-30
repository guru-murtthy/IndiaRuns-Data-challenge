import re
from typing import Dict, Any

SKILL_KEYWORDS = {
    "machine learning", "deep learning", "python", "faiss", "fastapi", "docker", 
    "lightgbm", "catboost", "embeddings", "retrieval", "ranking", "csv", "nlp", 
    "analytics", "react", "sql", "aws", "gcp", "azure", "kubernetes"
}

def parse_job_description(title: str, description: str) -> Dict[str, Any]:
    lowered = description.lower()
    skills = sorted({skill for skill in SKILL_KEYWORDS if skill in lowered})
    years_match = re.search(r"(\d+)\+?\s+years", lowered)
    min_experience = int(years_match.group(1)) if years_match else 2

    return {
        "title": title,
        "description": description,
        "required_skills": skills,
        "min_experience_years": min_experience,
    }
