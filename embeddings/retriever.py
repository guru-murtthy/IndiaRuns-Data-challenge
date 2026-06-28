import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

from embeddings.encoder import EmbeddingEncoder
from embeddings.faiss_builder import FaissBuilder
from embeddings.cache import load_candidate_ids
from embeddings.models import SearchResult, RetrievedCandidate

logger = logging.getLogger(__name__)

class CandidateRetriever:
    """
    Retrieves candidates using semantic search via FAISS and Sentence Transformers.
    """
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-large-en-v1.5",
        index_path: str = "models/candidate.index",
        ids_path: str = "models/candidate_ids.pkl"
    ):
        """
        Initializes the CandidateRetriever.
        
        Args:
            model_name: The HuggingFace model identifier for the encoder.
            index_path: Path to the FAISS index file.
            ids_path: Path to the pickled list of candidate IDs.
        """
        self.encoder = EmbeddingEncoder(model_name=model_name)
        
        index_file = Path(index_path)
        ids_file = Path(ids_path)
        
        logger.info(f"Loading FAISS index from {index_file}")
        self.index = FaissBuilder.load_index(index_file)
        
        logger.info(f"Loading candidate IDs from {ids_file}")
        self.candidate_ids = load_candidate_ids(ids_file)
        
        if self.index.ntotal != len(self.candidate_ids):
            raise ValueError(f"Mismatch between index size ({self.index.ntotal}) and number of IDs ({len(self.candidate_ids)})")
            
    def retrieve(self, jd_text: str, top_k: int = 500) -> SearchResult:
        """
        Retrieves the Top-K candidates for a given Job Description.
        
        Args:
            jd_text: The Job Description text.
            top_k: The number of candidates to retrieve.
            
        Returns:
            A SearchResult containing the retrieved candidates.
        """
        logger.info(f"Encoding Job Description for retrieval (Top-K={top_k})")
        # Encode JD and normalize
        jd_embedding = self.encoder.encode(jd_text, normalize=True)
        # Reshape to 2D array for FAISS
        jd_embedding = jd_embedding.reshape(1, -1)
        
        # Search FAISS
        logger.info("Searching FAISS index...")
        scores, indices = self.index.search(jd_embedding, top_k)
        
        # We only have one query, so take the first result
        scores = scores[0]
        indices = indices[0]
        
        retrieved_candidates = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx == -1:
                # FAISS returns -1 if it can't find enough neighbors
                break
                
            candidate_id = self.candidate_ids[idx]
            retrieved_candidates.append(
                RetrievedCandidate(
                    candidate_id=candidate_id,
                    score=float(score),
                    index=int(idx)
                )
            )
            
        logger.info(f"Retrieved {len(retrieved_candidates)} candidates.")
        
        return SearchResult(
            query=jd_text,
            candidates=retrieved_candidates
        )
