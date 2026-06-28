from pydantic import BaseModel, Field
from typing import List, Optional

class ExperienceItem(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    duration_months: int = 0
    is_current: bool

class CandidateProfile(BaseModel):
    # This maps the structure of your data lines safely
    experience: List[ExperienceItem] = Field(default_factory=list)
    skills: List[dict] = Field(default_factory=list)
    education: List[dict] = Field(default_factory=list)