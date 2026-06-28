import json
import os

def load_jd(file_path="data/raw/jd.json"):
    """
    Loads Job Description query from a JSON file.
    """
    default_jd = (
        "Senior ML Engineer Search Ranking Retrieval. "
        "Production experience with embeddings-based retrieval systems, sentence-transformers, vector search, FAISS, Pinecone, Weaviate, Qdrant, Milvus. "
        "Strong Python. Designing evaluation frameworks NDCG, MRR, MAP. 5-9 years experience, product companies. Pune, Noida."
    )
    if not os.path.exists(file_path):
        return default_jd
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("query", default_jd)
    except Exception:
        return default_jd
