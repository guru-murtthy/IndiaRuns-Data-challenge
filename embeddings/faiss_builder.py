# Placeholder for offline FAISS builder (this is meant to be run before rank.py)
import numpy as np

def build_faiss_index(embeddings: np.ndarray, index_path: str):
    """
    Offline process to build a FAISS index.
    In the E2E online ranker, this is bypassed as per constraints 
    (we use keyword or fast pre-computed heuristic).
    """
    pass

def load_faiss_index(index_path: str):
    """Load precomputed index."""
    return None
