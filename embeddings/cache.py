import logging
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def save_embeddings(embeddings: np.ndarray, file_path: Path) -> None:
    """
    Saves the embeddings array to a numpy file.
    
    Args:
        embeddings: The 2D NumPy array of embeddings.
        file_path: The path where to save the embeddings.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, embeddings)
        logger.info(f"Successfully saved embeddings to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save embeddings to {file_path}: {e}")
        raise

def load_embeddings(file_path: Path) -> np.ndarray:
    """
    Loads the embeddings array from a numpy file.
    
    Args:
        file_path: The path where the embeddings are saved.
        
    Returns:
        The 2D NumPy array of embeddings.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")
        
    try:
        embeddings = np.load(file_path)
        logger.info(f"Successfully loaded embeddings from {file_path}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings from {file_path}: {e}")
        raise

def save_candidate_ids(candidate_ids: List[str], file_path: Path) -> None:
    """
    Saves the list of candidate IDs to a pickle file.
    
    Args:
        candidate_ids: The list of candidate IDs.
        file_path: The path where to save the IDs.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(candidate_ids, f)
        logger.info(f"Successfully saved {len(candidate_ids)} candidate IDs to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save candidate IDs to {file_path}: {e}")
        raise

def load_candidate_ids(file_path: Path) -> List[str]:
    """
    Loads the list of candidate IDs from a pickle file.
    
    Args:
        file_path: The path where the IDs are saved.
        
    Returns:
        The list of candidate IDs.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Candidate IDs file not found: {file_path}")
        
    try:
        with open(file_path, 'rb') as f:
            candidate_ids = pickle.load(f)
        logger.info(f"Successfully loaded {len(candidate_ids)} candidate IDs from {file_path}")
        return candidate_ids
    except Exception as e:
        logger.error(f"Failed to load candidate IDs from {file_path}: {e}")
        raise
