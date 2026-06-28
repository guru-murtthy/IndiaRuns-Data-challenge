from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class Candidate(BaseModel):
    """
    Represents a candidate's profile data.
    """
    id: str
    title: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: Optional[str] = None
    projects: Optional[str] = None
    education: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class EmbeddingResult(BaseModel):
    """
    Represents the result of an embedding operation.
    """
    candidate_id: str
    embedding: Any # using Any because numpy array doesn't play perfectly with pydantic default validation without arbitrary_types_allowed
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RetrievedCandidate(BaseModel):
    """
    Represents a candidate retrieved from the index with its similarity score.
    """
    candidate_id: str
    score: float
    index: int


class SearchResult(BaseModel):
    """
    Represents the full result of a search query.
    """
    query: str
    candidates: List[RetrievedCandidate]


class EmbeddingCacheMetadata(BaseModel):
    """
    Metadata for the embedding cache.
    """
    model_name: str
    embedding_dim: int
    num_candidates: int
