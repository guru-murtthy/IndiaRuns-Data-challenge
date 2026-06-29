import logging
import time
import psutil
import os
import numpy as np

from ranking.feature_engineering import FeatureEngineer
from ranking.reranker import Reranker
from ranking.final_rank import FinalRankGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_dummy_data(num_candidates: int = 500):
    jd = {
        "title": "Senior Machine Learning Engineer",
        "experience": "5 years",
        "required_skills": ["Python", "Machine Learning", "LightGBM"],
        "preferred_skills": ["CatBoost", "Docker", "AWS"],
        "location": "Bangalore"
    }
    
    candidates = []
    for i in range(num_candidates):
        candidates.append({
            "candidate_id": f"CAND_{i:04d}",
            "title": "Machine Learning Engineer" if i % 2 == 0 else "Data Scientist",
            "experience": f"{np.random.randint(2, 8)} years",
            "skills": ["Python", "Machine Learning", "SQL", "Docker"] if i % 3 == 0 else ["Python", "CatBoost"],
            "company": "Tech Corp",
            "education": "M.Tech",
            "location": "Bangalore, India" if i % 2 == 0 else "Remote",
            "retrieval_score": np.random.uniform(0.7, 0.99)
        })
    return jd, candidates

def main():
    logger.info("Starting Ranker Benchmark...")
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024 # MB
    
    jd, candidates = generate_dummy_data(500)
    
    # 1. Initialize Reranker
    t0 = time.perf_counter()
    reranker = Reranker(model_type="fallback", calibration_method="min_max")
    t1 = time.perf_counter()
    init_time = (t1 - t0) * 1000
    
    # 2. Rerank (Feature Generation + Prediction + Calibration + Sort)
    t0 = time.perf_counter()
    reranked_candidates = reranker.rerank(jd, candidates)
    t1 = time.perf_counter()
    rerank_time = (t1 - t0) * 1000
    
    # 3. Final Rank
    t0 = time.perf_counter()
    final_list = FinalRankGenerator.generate(reranked_candidates)
    t1 = time.perf_counter()
    final_rank_time = (t1 - t0) * 1000
    
    mem_after = process.memory_info().rss / 1024 / 1024 # MB
    
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY (500 Candidates)")
    print("="*50)
    print(f"Initialization Time:      {init_time:.2f} ms")
    print(f"Reranking Latency:        {rerank_time:.2f} ms")
    print(f"Final Rank Generation:    {final_rank_time:.2f} ms")
    print(f"Total Pipeline Latency:   {init_time + rerank_time + final_rank_time:.2f} ms")
    print(f"Memory Usage Increase:    {mem_after - mem_before:.2f} MB")
    print("="*50)
    
    print(f"\nTop 3 Candidates:")
    for c in final_list[:3]:
        print(f"Rank {c.rank}: {c.candidate_id} (Score: {c.score:.4f})")

if __name__ == "__main__":
    main()
