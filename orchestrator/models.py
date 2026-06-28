from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobInput(BaseModel):
    title: str = Field(default="Untitled Role")
    description: str = Field(min_length=20)
    top_k: int = Field(default=100, ge=1, le=100)


class StageStatus(BaseModel):
    name: str
    status: str
    detail: str


class CandidateResult(BaseModel):
    rank: int
    candidate_id: str
    candidate_name: str
    title: str
    location: str
    semantic_score: float
    rank_score: float
    validation_score: float
    final_score: float
    reason: str
    flags: dict[str, Any]


class RunResult(BaseModel):
    run_id: str
    status: str
    job: JobInput
    stages: list[StageStatus]
    summary: dict[str, Any]
    results: list[CandidateResult]
