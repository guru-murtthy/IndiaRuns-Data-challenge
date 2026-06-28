import faiss
import numpy as np

def build_flat_ip_index(embeddings):
    """
    Builds a flat Inner Product (cosine similarity) FAISS index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    return index

def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    return faiss.read_index(file_path)
