import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1: First vector as a NumPy array.
        vec2: Second vector as a NumPy array.
        
    Returns:
        The cosine similarity score between -1.0 and 1.0.
    """
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)
        
    dot_prod = np.dot(vec1, vec2.T)
    norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    
    # Avoid division by zero
    norm_product = norm1 * norm2.T
    norm_product[norm_product == 0] = 1e-10
    
    return float(dot_prod / norm_product)

def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the dot product between two vectors.
    
    Args:
        vec1: First vector as a NumPy array.
        vec2: Second vector as a NumPy array.
        
    Returns:
        The dot product.
    """
    return float(np.dot(vec1.flatten(), vec2.flatten()))

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    L2 normalize a given vector.
    
    Args:
        vec: Vector or matrix of vectors to normalize.
        
    Returns:
        The normalized vector(s).
    """
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    else:
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        return vec / norms
