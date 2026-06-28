import logging
import argparse
from pathlib import Path

from embeddings.faiss_builder import FaissBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from cached embeddings.")
    parser.add_argument("--embeddings_path", type=str, default="models/candidate_embeddings.npy", help="Path to input embeddings numpy file.")
    parser.add_argument("--index_path", type=str, default="models/candidate.index", help="Path to output FAISS index file.")
    
    args = parser.parse_args()
    
    embeddings_file = Path(args.embeddings_path)
    index_file = Path(args.index_path)
    
    if not embeddings_file.exists():
        logger.error(f"Embeddings file not found: {embeddings_file}")
        logger.error("Please run generate_embeddings.py first.")
        return
        
    logger.info("Starting FAISS index build process...")
    FaissBuilder.build_and_save_from_cache(embeddings_file, index_file)
    logger.info("FAISS index build process completed.")

if __name__ == "__main__":
    main()
