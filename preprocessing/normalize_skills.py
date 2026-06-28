from .clean_text import clean_text

def normalize_skill(skill_name):
    """
    Normalizes skill names to standard forms.
    """
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
