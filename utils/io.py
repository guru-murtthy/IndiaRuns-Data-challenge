import json
from pathlib import Path
from typing import Any, List, Dict

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            data.append(json.loads(stripped))
    return data

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
