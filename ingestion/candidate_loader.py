import json
import os

def load_candidates(file_path):
    """
    Loads candidates from JSON or JSONL formats.
    """
    if not os.path.exists(file_path):
        # Fallback to current directory if not found at specific path
        base_name = os.path.basename(file_path)
        if os.path.exists(base_name):
            file_path = base_name
        else:
            raise FileNotFoundError(f"Candidate dataset not found at {file_path}")

    candidates = []
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    return candidates
