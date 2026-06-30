from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

CANDIDATES_FILE = RAW_DATA_DIR / "candidates.jsonl"
JD_FILE = RAW_DATA_DIR / "jd.json"

# Pipeline constants
TOP_K_DEFAULT = 100
MAX_MEMORY_GB = 16
MAX_RUNTIME_MINUTES = 5

# Schema constants
REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
