# Redrob Candidate Discovery & Ranking Pipeline (IndiaRuns Data Challenge)

This repository contains the end-to-end intelligent candidate discovery and ranking system developed for the **Redrob Hackathon v4**. The system is built to process a pool of 100,000 candidates and retrieve the top 100 best-fit candidates for a **Senior AI Engineer (Founding Team)** role, meeting all validation, compute, and performance constraints.

## Architecture Overview

The ranking system consists of two primary phases: an offline precomputation phase and an online ranking/validation phase.

```
                  +--------------------------------+
                  |        Job Description         |
                  +---------------+----------------+
                                  |
                                  v
                  +---------------+----------------+
                  |       JD Query Encoder         |
                  +---------------+----------------+
                                  |
                                  v
                  +---------------+----------------+
                  |      FAISS Semantic Search     |
                  |     (Retrieves Top 1000)       |
                  +---------------+----------------+
                                  |
                                  v
                  +---------------+----------------+
                  |   Chronological Validation     |
                  |     (Honeypot Filter)          |
                  +---------------+----------------+
                                  |
                                  v
                  +---------------+----------------+
                  |      LightGBM LTR Model        |
                  |    + Custom Feature Scoring    |
                  +---------------+----------------+
                                  |
                                  v
                  +---------------+----------------+
                  |  Post-Retrieval Title Filter   |
                  +---------------+----------------+
                                  |
                                  v
                  +---------------+----------------+
                  |     Deterministic Sorting      |
                  |     (Score Desc, ID Asc)       |
                  +---------------+----------------+
                                  |
                                  v
                  +---------------+----------------+
                  |  Reasoning Engine Generator    |
                  +---------------+----------------+
                                  |
                                  v
                  +--------------------------------+
                  |         Submission CSV         |
                  +--------------------------------+
```

---

## Member 3 Features & Deliverables

As part of the **Member 3** deliverables, this codebase implements a robust validation and scoring layer designed to filter out synthetic honeypots and evaluate candidate profile quality:

### 1. Chronological Timeline Validation & Honeypot Detection
Our validation layer detects and filters out synthetic "honeypot" profiles using several strict physical and chronological checks:
- **Startup Founding Constraints**: Checks if the candidate claims to have worked at a startup (e.g., Krutrim, Sarvam AI, CRED) before its founding year, or if their reported duration exceeds the maximum possible age of the startup.
- **Skill Duration Discrepancy**: Flags keyword-stuffer profiles listing "expert" or "advanced" skills with a duration of 0 months.
- **Years of Experience vs. Elapsed Time**: Computes the elapsed time since the candidate's first reported job and flags profiles where the claimed Years of Experience exceeds this physical limit.
- **Senior Role College Check**: Flags candidates claiming senior/lead roles (e.g., Project Manager, Lead Engineer) years before graduating college.
- **Overlapping Jobs**: Flags profiles with multiple concurrent full-time job overlaps.

### 2. Career Growth & Stability Scoring
Evaluates the candidate's professional trajectory and job stability:
- **Tenure Rewards**: Grants score boosts for candidates with proven long-term stays (>=3 years) at product/startup companies.
- **Job-Hopping Penalties**: Multiplies the score with a penalty factor for excessive short-term stays (<12 months) or a low average job duration (<18 months).
- **Service-to-Product Jumps**: Computes trajectory points for climbing tiers (e.g., moving from consulting firms like TCS/Infosys to product startups).

### 3. Skill Evidence Score
Weighs skill proficiencies against their duration and peer endorsements to identify genuine technical experience and penalize keyword-stuffing.

### 4. Factual, Deterministic Reasoning Engine
Generates 1-2 sentence justifications that highlight candidate-specific facts (e.g., exact years of experience, current company, specific matching skills, and notice period). The generator uses a hash of the candidate ID to select sentence clauses, guaranteeing **100% deterministic and reproducible reasoning** across runs.

---

## Project Structure

- `member3/`: Core package containing validation, scoring, and reasoning logic.
  - [timeline_validation.py](file:///home/gururaj/Videos/india%20RUns/member3/timeline_validation.py): Chronological and startup age validation (honeypot detection).
  - [scoring_features.py](file:///home/gururaj/Videos/india%20RUns/member3/scoring_features.py): Skill evidence, career growth, and resume authenticity scores.
  - [reasoning_engine.py](file:///home/gururaj/Videos/india%20RUns/member3/reasoning_engine.py): Factual reasoning generation.
- [precompute.py](file:///home/gururaj/Videos/india%20RUns/precompute.py): Precomputation script to build index and train the LightGBM model.
- [rank.py](file:///home/gururaj/Videos/india%20RUns/rank.py): Main ranking script.
- [utils.py](file:///home/gururaj/Videos/india%20RUns/utils.py): Utility functions for text cleaning and normalizations.
- `requirements.txt`: Python package dependencies.
- `submission_metadata.yaml`: Metadata file mirroring the portal submission settings.

---

## Setup & Execution

### 1. Installation
Install the required packages in your Python environment:
```bash
pip install -r requirements.txt
```

### 2. Precomputation (Offline)
This step processes the full candidate dataset, filters out honeypots, extracts features, generates text embeddings using `all-MiniLM-L6-v2`, builds a FAISS index, and trains the LightGBM ranking model. 
```bash
python precompute.py --candidates candidates.jsonl
```
*Note: This generates `faiss_index.bin`, `candidate_features.csv`, `lightgbm_model.bin`, and `candidate_ids.json`.*

### 3. Running the Ranker (Sandboxed Execution)
This is the official reproduction command that runs within the 5-minute wall-clock constraint on CPU without network:
```bash
python rank.py --candidates candidates.jsonl --out team_xxx.csv
```

### 4. Format Validation
Verify that your submission CSV is 100% compliant with the challenge rules before uploading:
```bash
python validate_submission.py team_xxx.csv
```