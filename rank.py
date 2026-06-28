#!/usr/bin/env python3
"""
CLI Entry Point for the Pipeline.
Reproducibility constraint: Must support `--candidates` and `--out`
Must complete within 5 minutes on a 16GB CPU without external APIs.
"""

import argparse
import sys
from pathlib import Path

from orchestrator.models import JobInput
from orchestrator.services.orchestrator import run_pipeline
from orchestrator.services.export_service import export_run_to_csv

def main():
    parser = argparse.ArgumentParser(description="India Runs Data AI Challenge - CLI Entry Point")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl file")
    parser.add_argument("--out", type=str, required=True, help="Path to output submission.csv")
    parser.add_argument("--jd", type=str, default=None, help="Path to Job Description JSON file (optional)")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top candidates to retrieve")
    args = parser.parse_args()

    jd_text = "Default AI Engineer Role"
    if args.jd:
        print(f"Loading Job Description from {args.jd}...")
        try:
            jd_path = Path(args.jd)
            jd_text = jd_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading JD: {e}")
            sys.exit(1)

    print(f"Loading candidates from {args.candidates}...")
    job = JobInput(title="CLI Job", description=jd_text, top_k=args.top_k)
    
    print("Running end-to-end pipeline...")
    # Pass candidates_path to the orchestrator to mock or parse
    run = run_pipeline(job, candidates_path=Path(args.candidates))
    
    print(f"Pipeline completed with {len(run.results)} candidates.")

    # Export to the required `--out` path
    output_path = export_run_to_csv(run, custom_out_path=Path(args.out))
    print(f"Results exported successfully to {output_path}")

if __name__ == "__main__":
    main()
