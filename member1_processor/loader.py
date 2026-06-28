import json
import os
from typing import Generator

# Fallback mechanism: handles both direct standalone runs and package imports safely
try:
    from .schemas import CandidateProfile
except ImportError:
    from schemas import CandidateProfile


def stream_candidates(file_name: str = "candidates.jsonl") -> Generator[CandidateProfile, None, None]:
    # Look one folder level up to find the massive 487MB data asset safely
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, file_name)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield CandidateProfile(**json.loads(line))
    except FileNotFoundError:
        print(f"❌ Could not locate the dataset file at: {file_path}")


if __name__ == "__main__":
    print("⏳ Testing localized streaming architecture...")
    try:
        stream = stream_candidates("candidates.jsonl")
        for candidate in stream:
            print("🚀 [Module Test] Isolation verification: SUCCESS.")
            print(f"Extracted sample profile. Total experience items: {len(candidate.experience)}")
            break
    except Exception as e:
        print(f"❌ Test failed due to error: {e}")