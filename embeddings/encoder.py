import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from embeddings.similarity import normalize_vector

logger = logging.getLogger(__name__)

class EmbeddingEncoder:
    """
    Wrapper class around Sentence Transformers for generating semantic embeddings.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initializes the EmbeddingEncoder with a specified model.
        
        Args:
            model_name: The HuggingFace model identifier. Defaults to 'BAAI/bge-large-en-v1.5'.
        """
        logger.info(f"Loading embedding model: {model_name} on CPU")
        self.model_name = model_name
        # Force CPU usage as requested
        self.model = SentenceTransformer(model_name, device="cpu")
        
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encodes a single text string into an embedding vector.
        
        Args:
            text: The input text to encode.
            normalize: Whether to L2 normalize the resulting vector.
            
        Returns:
            A NumPy array containing the embedding.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if normalize:
            embedding = normalize_vector(embedding)
            
        return embedding

    def encode_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encodes a list of text strings into embedding vectors.
        
        Args:
            texts: List of input texts to encode.
            batch_size: Number of texts to process in parallel.
            normalize: Whether to L2 normalize the resulting vectors.
            
        Returns:
            A 2D NumPy array where each row represents the embedding of the corresponding text.
        """
        logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
        
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        if normalize:
            embeddings = normalize_vector(embeddings)
            
        return embeddings
