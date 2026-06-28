# Redrob Hackathon — Intelligent Candidate Discovery
> **Team Member 4 Repository** · India Runs Data & AI Challenge

This repository implements the **Member 4 deliverables**: the end-to-end pipeline orchestrator, FastAPI dashboard, and compliant CSV exporter. It is designed to work with the outputs of Members 1, 2, and 3 once they are ready.

---

## Project Structure

```text
├── rank.py                   # Stage 3 evaluation entry point (CLI)
├── submission_metadata.yaml  # Required portal metadata
├── requirements.txt
│
├── orchestrator/                      # Member 4: FastAPI Dashboard & Orchestration
│   ├── main.py               # Routes (/, /runs, /runs/{id}/export)
│   ├── models.py             # Pydantic data models
│   ├── adapters/             # Integration points for Members 1, 2, 3
│   │   ├── member1_adapter.py    # JD parsing + candidate ingestion
│   │   ├── member2_adapter.py    # Retrieval + heuristic ranking (replace with ML)
│   │   └── member3_adapter.py    # Validation + honeypot detection + reasoning
│   ├── services/
│   │   ├── orchestrator.py   # Pipeline sequencer
│   │   └── export_service.py # 100-row compliant CSV generator
│   ├── templates/            # Jinja2 HTML templates
│   └── static/               # CSS (glassmorphic dark theme)
│
├── modules/                  # Placeholder modules for team integration
│   ├── member1_parsing/      # ingestion, parser, preprocessing
│   ├── member2_ranking/      # embeddings, feature_engineering, ranking
│   └── member3_validation/   # validation, reasoning, evaluation
│
├── docs/
│   ├── architecture.md       # System architecture diagram & component breakdown
│   └── integration_guide.md  # Step-by-step guide for team members to plug in
│
├── data/
│   ├── sample/               # Small sample for fast local testing
│   ├── raw/                  # Place the real candidates.jsonl here
│   └── output/               # Generated submission.csv (latest run)
│
├── submission/
│   └── outputs/              # All generated CSV files (keyed by run-id)
│
├── scripts/
│   └── validate_submission.py  # Official challenge validator
│
└── tests/                    # Basic endpoint tests
```

---

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Run the CLI pipeline (Stage 3 command)
```bash
python rank.py --candidates data/raw/candidates.jsonl --out submission/outputs/submission.csv
```

### 3. Validate the output
```bash
python scripts/validate_submission.py submission/outputs/submission.csv
```

### 4. Launch the Web Dashboard
```bash
uvicorn orchestrator.main:app --reload
```
Open `http://127.0.0.1:8000` in your browser.

### 5. Docker
```bash
docker compose up --build
```

---

## For Team Members: How to Integrate Your Work

1. See **[`docs/integration_guide.md`](docs/integration_guide.md)** for the exact function signatures to implement.
2. Drop your ML code into `orchestrator/adapters/member2_adapter.py` (ranker) and `orchestrator/adapters/member3_adapter.py` (validator).
3. Run the E2E test with `python rank.py --candidates ...` to verify the CSV passes `validate_submission.py`.

---

## Critical Constraints (from submission_spec.md)
> [!CAUTION]
> The ranking step that produces the CSV **must NOT**:
> - Call any hosted LLM APIs (OpenAI, Anthropic, Cohere, etc.)
> - Use GPUs
> - Exceed 5 minutes on a 16 GB RAM CPU-only machine
> - Make any network requests

All pre-computation (embeddings, FAISS index building) must be done as a **separate offline step** before the `rank.py` command is run.

---

## Architecture Overview
See [`docs/architecture.md`](docs/architecture.md) for the full pipeline diagram.

## Sandbox
This FastAPI app can be deployed to HuggingFace Spaces (Docker), Replit, or Streamlit Cloud to fulfil the mandatory sandbox requirement. See [`docs/integration_guide.md`](docs/integration_guide.md) for instructions.
