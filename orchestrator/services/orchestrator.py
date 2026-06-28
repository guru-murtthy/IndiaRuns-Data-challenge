from __future__ import annotations

from uuid import uuid4

from orchestrator.adapters.member1_adapter import load_candidates, parse_job_description
from orchestrator.adapters.member2_adapter import rank_candidates
from orchestrator.adapters.member3_adapter import validate_and_explain
from orchestrator.models import CandidateResult, JobInput, RunResult, StageStatus


def run_pipeline(job: JobInput, candidates_path=None) -> RunResult:
    stages = [
        StageStatus(name="JD Parsing", status="running", detail="Extracting requirements"),
        StageStatus(name="Candidate Loading", status="pending", detail="Waiting to load candidates"),
        StageStatus(name="Retrieval + Ranking", status="pending", detail="Waiting to score candidates"),
        StageStatus(name="Validation + Reasoning", status="pending", detail="Waiting to validate results"),
    ]

    parsed_jd = parse_job_description(job.title, job.description)
    stages[0] = StageStatus(name="JD Parsing", status="complete", detail="Structured JD object ready")

    candidates = load_candidates(candidates_path)
    stages[1] = StageStatus(name="Candidate Loading", status="complete", detail=f"Loaded {len(candidates)} candidates")

    ranked_candidates = rank_candidates(parsed_jd, candidates, job.top_k)
    stages[2] = StageStatus(
        name="Retrieval + Ranking",
        status="complete",
        detail=f"Generated top {len(ranked_candidates)} candidates",
    )

    validated_candidates = validate_and_explain(parsed_jd, ranked_candidates)
    stages[3] = StageStatus(
        name="Validation + Reasoning",
        status="complete",
        detail="Validation flags and candidate reasons generated",
    )

    results = [
        CandidateResult(
            rank=index,
            candidate_id=item["candidate_id"],
            candidate_name=item["candidate_name"],
            title=item["title"],
            location=item["location"],
            semantic_score=item["semantic_score"],
            rank_score=item["rank_score"],
            validation_score=item["validation_score"],
            final_score=item["final_score"],
            reason=item["reason"],
            flags=item["flags"],
        )
        for index, item in enumerate(validated_candidates, start=1)
    ]

    summary = {
        "required_skills": parsed_jd["required_skills"],
        "min_experience_years": parsed_jd["min_experience_years"],
        "candidate_count": len(candidates),
        "top_score": results[0].final_score if results else 0,
        "output_columns": ["candidate_id", "rank", "score", "reasoning"],
    }

    return RunResult(
        run_id=f"run-{uuid4().hex[:8]}",
        status="complete",
        job=job,
        stages=stages,
        summary=summary,
        results=results,
    )
