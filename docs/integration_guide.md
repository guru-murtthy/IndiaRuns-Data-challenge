# Integration Guide for Members 1, 2, and 3

## Overview
Member 4's FastAPI orchestrator is the central hub. It calls your code through **adapter files** located at `orchestrator/adapters/`. Each adapter has a **well-defined function signature** you must match. Do not change the function names or argument types — just replace the body with your real logic.

---

## Member 1: Ingestion & JD Parsing
### File: `orchestrator/adapters/member1_adapter.py`

### What is currently there
- `parse_job_description(title, description)` — parses the JD text into skills and experience requirements.
- `load_candidates(candidate_path)` — loads the JSONL file and normalizes each candidate to a flat dict.

### What you need to replace / upgrade
1. **`parse_job_description`**: Replace the keyword-extraction heuristic with your proper NLP parser. The output dict MUST contain these keys:
   ```python
   {
       "title": str,
       "description": str,
       "required_skills": list[str],        # lowercase skill names
       "min_experience_years": int
   }
   ```
2. **`load_candidates`**: Current implementation already handles both JSON and JSONL format using the official schema. You can upgrade the normalization to add more parsed fields if needed. But keep all the current keys intact.

---

## Member 2: Retrieval & Ranking
### File: `orchestrator/adapters/member2_adapter.py`

### Function signature (do not change this)
```python
def rank_candidates(
    jd: dict[str, object],
    candidates: list[dict[str, object]],
    top_k: int
) -> list[dict[str, object]]:
```

### What you must return
A list of candidate dicts, limited to `top_k`, **sorted by best match first**. Each dict must contain all original candidate keys PLUS:
```python
{
    "semantic_score": float,     # 0.0 to 1.0
    "rank_score": float,         # 0.0 to 1.0
}
```

### Compute constraints reminder
- **No GPU** during the ranking phase.
- **No external APIs** (OpenAI, Cohere, etc.) during ranking.
- **Must complete in under 5 minutes** on a 16GB CPU machine.
- Pre-compute your FAISS index or embeddings offline (that step can be a separate script).

---

## Member 3: Validation & Reasoning
### File: `orchestrator/adapters/member3_adapter.py`

### Function signature (do not change this)
```python
def validate_and_explain(
    jd: dict[str, object],
    ranked_candidates: list[dict[str, object]]
) -> list[dict[str, object]]:
```

### What you must return
The same candidate list but with these additional keys on every dict:
```python
{
    "validation_score": float,   # 0.0 to 1.0
    "final_score": float,        # 0.0 to 1.0 — used for CSV export
    "reason": str,               # 1-2 sentences, specific to this candidate (see below)
    "flags": dict,               # any validation flags you generated
}
```

### Reasoning quality requirements (from submission_spec.md)
- **DO NOT** use identical reasoning strings for all candidates.
- **DO NOT** mention skills not in the candidate's profile.
- **DO** write specific, honest reasoning tied to the candidate's actual data.
- This is sampled at Stage 4 — bad reasoning = score penalty.

---

## Data Contract: Candidate Dict Schema
The canonical dict passed between adapters looks like this:
```python
{
    "candidate_id": "CAND_0000001",   # from real data
    "candidate_name": str,
    "title": str,                     # current_title from profile
    "location": str,
    "skills": list[str],              # lowercase skill names
    "experience_years": float,
    "summary": str,
    "response_rate": float,           # from redrob_signals
    # After Member 2 adds:
    "semantic_score": float,
    "rank_score": float,
    # After Member 3 adds:
    "validation_score": float,
    "final_score": float,
    "reason": str,
    "flags": dict,
}
```
