import pytest
import numpy as np
from embeddings.encoder import EmbeddingEncoder

class TestEmbeddingEncoder:
    
    @pytest.fixture(scope="class")
    def encoder(self):
        # We can use a smaller/faster model for unit tests to speed them up
        return EmbeddingEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_encode_single(self, encoder):
        text = "Senior Software Engineer with Python skills."
        embedding = encoder.encode(text, normalize=True)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        # Check normalization (L2 norm should be approx 1.0)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0)
        
    def test_encode_batch(self, encoder):
        texts = [
            "Senior Backend Engineer",
            "Data Scientist specializing in deep learning",
            "Frontend developer with React experience"
        ]
        
        embeddings = encoder.encode_batch(texts, batch_size=2, normalize=True)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == 3
        
        # Check normalization for each vector
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0)
