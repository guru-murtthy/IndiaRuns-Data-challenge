import re
from member3 import (
    validate_candidate_timeline,
    calculate_skill_evidence_score,
    calculate_career_growth_score,
    calculate_resume_authenticity_score
)

CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini", "HCL", "Tech Mahindra", "Mphasis"
}

def clean_text(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())

def normalize_skill(skill_name):
    name = clean_text(skill_name)
    mappings = {
        "sentence transformer": "sentence-transformers",
        "sentence transformers": "sentence-transformers",
        "vector database": "vector search",
        "vector db": "vector search",
        "embeddings-based retrieval": "embeddings",
        "embedding": "embeddings",
        "learning to rank": "learning-to-rank",
        "ltr": "learning-to-rank",
        "xgb": "xgboost",
        "lgbm": "lightgbm",
        "langchain": "langchain",
        "open ai": "openai",
        "natural language processing": "nlp"
    }
    return mappings.get(name, name)

def normalize_title(title_name):
    title = clean_text(title_name)
    if "backend" in title:
        return "backend engineer"
    if "machine learning" in title or " ml " in title or title.startswith("ml ") or title.endswith(" ml") or "ai engineer" in title or "artificial intelligence" in title:
        return "ml engineer"
    if "data engineer" in title:
        return "data engineer"
    if "operations" in title:
        return "operations manager"
    if "marketing" in title:
        return "marketing manager"
    if "support" in title or "helpdesk" in title:
        return "customer support"
    if "manager" in title or "director" in title or "lead" in title or "head" in title:
        return "lead/manager"
    return "other"
