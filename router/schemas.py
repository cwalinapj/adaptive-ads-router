"""Pydantic schemas for the Adaptive Ads Router."""

from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List, Literal


class SessionNeuralState(BaseModel):
    session_id: str
    site_id: str
    page_id: str
    llm_provider: str
    timestamp: str
    dwell_time: float = Field(ge=0)
    max_scroll: float = Field(ge=0, le=1)
    interaction_count: int = Field(ge=0)
    cta_intent_score: float = Field(ge=0, le=1)
    conversion: bool = False
    revenue_value: float = Field(ge=0, default=0.0)


class TombstoneRecord(BaseModel):
    tombstone_id: str
    site_id: str
    terminated_page_id: str
    successor_page_id: str
    final_divergence_gap: float
    total_lifetime_events: int
    primary_failure_mode: Literal[
        "low_conversion", "high_bounce", "low_engagement",
        "slow_load", "diff_violation", "manual_kill"
    ]
    llm_provider: str
    hypothesis_was: str
    actual_result: str
    created_at: Optional[str] = None


class RouteRequest(BaseModel):
    visitor_id: str
    device_type: Literal["mobile", "desktop", "tablet"] = "desktop"
    referrer: Optional[str] = None
    utm_source: Optional[str] = None
    utm_campaign: Optional[str] = None


class RouteResponse(BaseModel):
    site_id: str
    page_id: str
    container_url: str
    regime: Literal["first_100", "middle", "neural"]
    arm_index: int
    session_id: str


class OutcomeRequest(BaseModel):
    site_id: str
    page_id: str
    session_id: str
    converted: bool
    revenue: float = 0.0
    dwell_time: Optional[float] = None
    max_scroll: Optional[float] = None
    interaction_count: Optional[int] = None


class OutcomeResponse(BaseModel):
    recorded: bool
    regime: str
    winner_declared: bool = False
    winner_page_id: Optional[str] = None
    should_regenerate: bool = False
    loser_page_id: Optional[str] = None


class MutationDirection(BaseModel):
    mutation_id: str
    site_id: str
    source_page_id: str
    target_page_id: str
    changes: Dict[str, Any] = {}
    diff_score: float = Field(ge=0, le=1)
    hypothesis: str
    expected_impact: Literal["positive", "negative", "neutral"]


class DiffValidationRequest(BaseModel):
    site_id: str
    page_id: str
    proposed_diff: MutationDirection


class DiffValidationResponse(BaseModel):
    allowed: bool
    regime: str
    diff_score: float
    max_allowed: float
    violations: List[str] = []
    reason: Optional[str] = None
