# Core JD skills to check for matches
CORE_JD_SKILLS = {
    "embeddings", "retrieval", "vector search", "faiss", "pinecone", 
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", 
    "python", "sentence-transformers", "ndcg", "mrr", "map",
    "fine-tuning", "lora", "qlora", "peft", "learning-to-rank", 
    "xgboost", "lightgbm", "catboost", "distributed systems", "nlp"
}

def parse_jd_skills(jd_text):
    """
    Parses and extracts key skills from Job Description text.
    """
    matched = []
    text_lower = jd_text.lower()
    for skill in CORE_JD_SKILLS:
        if skill in text_lower:
            matched.append(skill)
    return matched
