import logging
import argparse
from pathlib import Path
import json

from embeddings.generator import EmbeddingGenerator

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for candidates.")
    # Assuming a JSON file containing parsed candidates for now.
    # We could also use the actual source (like a database or another file).
    # Since the request just says "Read candidate list", we will accept a JSON file
    # or generate some dummy data for testing purposes if no file is provided.
    parser.add_argument("--input", type=str, help="Path to input JSON containing a list of candidates.")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save embeddings.")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5", help="SentenceTransformer model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    
    args = parser.parse_args()
    
    candidates = []
    if args.input:
        logger.info(f"Loading candidates from {args.input}")
        with open(args.input, 'r') as f:
            candidates = json.load(f)
    else:
        logger.info("No input provided. Generating sample dummy candidate data for testing.")
        # Dummy data matching the structure described in CandidateTextBuilder
        candidates = [
            {
                "id": "c1",
                "title": "Senior Backend Engineer",
                "skills": ["Python", "FastAPI", "Docker"],
                "experience": "6 years",
                "projects": "Built scalable APIs",
                "education": "B.Tech Computer Science",
                "location": "Bangalore",
                "company": "Google"
            },
            {
                "id": "c2",
                "title": "Machine Learning Engineer",
                "skills": ["Python", "PyTorch", "TensorFlow"],
                "experience": "4 years",
                "projects": "Recommendation System",
                "education": "M.Sc Artificial Intelligence",
                "location": "Hyderabad",
                "company": "Microsoft"
            }
        ]
        
    generator = EmbeddingGenerator(model_name=args.model_name, batch_size=args.batch_size)
    generator.generate_and_save(candidates, Path(args.output_dir))

if __name__ == "__main__":
    main()
