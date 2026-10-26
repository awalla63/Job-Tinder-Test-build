"""Request and response models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CompleteProfileIn(BaseModel):
    role: Literal["seeker", "employer"]
    display_name: str = Field(default="", max_length=120)
    company_name: str | None = Field(default=None, max_length=200)


class SurveyIn(BaseModel):
    """Preferences. Every field is optional and None means 'no preference'.

    These become hard filters in ranking.py, never model features -- the labelled
    training data has no salary/location/work-type columns, so anything derived from
    them could not be learned. See the note at the top of ml/features.py.
    """
    min_salary: float | None = None
    remote_pref: Literal["remote", "onsite", "any"] | None = None
    work_types: list[str] | None = None
    states: list[str] | None = None
    categories: list[str] | None = None
    experience_levels: list[str] | None = None
    answers: dict = Field(default_factory=dict)


class JobCard(BaseModel):
    id: str
    title: str
    company_name: str
    location: str | None = None
    description: str
    normalized_salary: float | None = None
    pay_period: str | None = None
    formatted_work_type: str | None = None
    formatted_experience_level: str | None = None
    remote_allowed: bool = False
    application_url: str | None = None
    source: str
    score: float
    reasons: list[str] = Field(default_factory=list)


class CandidateCard(BaseModel):
    id: str
    display_name: str
    headline: str | None = None
    resume_preview: str
    years_experience: float | None = None
    predicted_category: str | None = None
    is_demo: bool = False
    score: float
    reasons: list[str] = Field(default_factory=list)


class SwipeIn(BaseModel):
    target_type: Literal["job", "seeker"]
    target_id: str
    direction: Literal["like", "pass"]
    score_at_swipe: float | None = None


class SwipeOut(BaseModel):
    recorded: bool
    matched: bool = False
    match_id: str | None = None
    application_url: str | None = None


class JobIn(BaseModel):
    title: str = Field(min_length=2, max_length=300)
    description: str = Field(min_length=50)
    company_name: str | None = None
    location: str | None = None
    state: str | None = None
    normalized_salary: float | None = None
    pay_period: str | None = None
    formatted_work_type: str | None = None
    formatted_experience_level: str | None = None
    remote_allowed: bool = False
    application_url: str | None = None


class MatchOut(BaseModel):
    id: str
    job_id: str
    job_title: str
    company_name: str
    application_url: str | None = None
    source: str
    counterpart_name: str | None = None
    created_at: str
