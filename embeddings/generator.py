import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from embeddings.encoder import EmbeddingEncoder
from embeddings.text_builder import CandidateTextBuilder
from embeddings.cache import save_embeddings, save_candidate_ids

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Coordinates the generation of embeddings from raw candidate data.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 32):
        """
        Initializes the EmbeddingGenerator.
        
        Args:
            model_name: The HuggingFace model identifier for the encoder.
            batch_size: The batch size for processing candidates.
        """
        self.encoder = EmbeddingEncoder(model_name=model_name)
        self.batch_size = batch_size
        
    def generate_and_save(
        self, 
        candidates: List[Dict[str, Any]], 
        output_dir: Path,
        embeddings_filename: str = "candidate_embeddings.npy",
        ids_filename: str = "candidate_ids.pkl"
    ) -> None:
        """
        Generates embeddings for a list of candidates and saves them to disk.
        
        Args:
            candidates: A list of candidate dictionaries.
            output_dir: The directory where to save the outputs.
            embeddings_filename: The name of the file to save the numpy embeddings.
            ids_filename: The name of the file to save the candidate IDs.
        """
        if not candidates:
            logger.warning("No candidates provided for embedding generation.")
            return
            
        logger.info(f"Starting embedding generation for {len(candidates)} candidates.")
        
        candidate_ids = []
        texts = []
        
        for candidate in candidates:
            # Assumes every candidate has an 'id' field
            if 'id' not in candidate:
                logger.warning(f"Skipping candidate without id: {candidate}")
                continue
                
            candidate_ids.append(str(candidate['id']))
            text = CandidateTextBuilder.build_text(candidate)
            texts.append(text)
            
        if not texts:
            logger.error("No valid texts built from candidates.")
            return
            
        logger.info("Encoding texts into embeddings...")
        embeddings = self.encoder.encode_batch(texts, batch_size=self.batch_size, normalize=True)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings_path = output_dir / embeddings_filename
        ids_path = output_dir / ids_filename
        
        logger.info(f"Saving embeddings and IDs to {output_dir}")
        save_embeddings(embeddings, embeddings_path)
        save_candidate_ids(candidate_ids, ids_path)
        
        logger.info("Embedding generation and saving completed successfully.")
