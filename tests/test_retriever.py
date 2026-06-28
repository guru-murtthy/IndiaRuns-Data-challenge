import pytest
import numpy as np
import faiss
import tempfile
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

from embeddings.retriever import CandidateRetriever
from embeddings.encoder import EmbeddingEncoder

class TestCandidateRetriever:

    @pytest.fixture
    def mock_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create dummy embeddings (e.g., 5 candidates, dimension 384)
            dim = 384
            num_candidates = 5
            embeddings = np.random.rand(num_candidates, dim).astype('float32')
            faiss.normalize_L2(embeddings)
            
            # Create and save index
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            index_path = tmpdir_path / "test.index"
            faiss.write_index(index, str(index_path))
            
            # Create and save candidate IDs
            candidate_ids = [f"c{i}" for i in range(num_candidates)]
            ids_path = tmpdir_path / "test_ids.pkl"
            with open(ids_path, 'wb') as f:
                pickle.dump(candidate_ids, f)
                
            yield index_path, ids_path, dim

    @patch('embeddings.retriever.EmbeddingEncoder')
    def test_retrieve(self, MockEncoder, mock_environment):
        index_path, ids_path, dim = mock_environment
        
        # Setup mock encoder
        mock_encoder_instance = MagicMock()
        MockEncoder.return_value = mock_encoder_instance
        # Mock encoding a query (returns normalized vector)
        mock_query_emb = np.random.rand(dim).astype('float32')
        mock_query_emb = mock_query_emb / np.linalg.norm(mock_query_emb)
        mock_encoder_instance.encode.return_value = mock_query_emb
        
        retriever = CandidateRetriever(
            model_name="dummy_model", 
            index_path=str(index_path), 
            ids_path=str(ids_path)
        )
        
        results = retriever.retrieve("looking for python dev", top_k=3)
        
        assert results.query == "looking for python dev"
        assert len(results.candidates) == 3
        
        for i, candidate in enumerate(results.candidates):
            assert isinstance(candidate.candidate_id, str)
            assert candidate.candidate_id.startswith("c")
            assert isinstance(candidate.score, float)
            assert isinstance(candidate.index, int)
            
            # Scores should be sorted descending
            if i > 0:
                assert results.candidates[i-1].score >= candidate.score
