from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ExperienceItem(BaseModel):
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: Optional[str] = None
    duration_months: int = 0
    is_current: bool = False
    company_size: str = ""

class SkillItem(BaseModel):
    name: str
    proficiency: str = "beginner"
    duration_months: int = 0
    endorsements: int = 0

class RedrobSignals(BaseModel):
    recruiter_response_rate: float = 0.5
    open_to_work_flag: bool = False
    github_activity_score: float = -1.0
    profile_completeness_score: float = 0.0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    notice_period_days: int = 0

class CandidateProfile(BaseModel):
    candidate_id: str
    experience: List[ExperienceItem] = Field(default_factory=list)
    skills: List[SkillItem] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    signals: RedrobSignals = Field(default_factory=RedrobSignals)
    raw_record: Dict[str, Any] = Field(default_factory=dict)
    
    # Computed basic fields
    candidate_name: str = ""
    title: str = ""
    location: str = ""
    experience_years: float = 0.0
    summary: str = ""
