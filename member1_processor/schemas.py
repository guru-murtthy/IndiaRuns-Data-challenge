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

class JobDescriptionSchema(BaseModel):
    jd_id: str = "unknown"
    required_skills: List[str] = Field(default_factory=list)
    min_experience_months: int = 0
    preferred_titles: List[str] = Field(default_factory=list)
    location_preference: str = "any"