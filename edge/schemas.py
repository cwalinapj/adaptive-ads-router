"""Schemas for Edge Composer Service."""

from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List, Literal
from datetime import datetime


class VisitorContext(BaseModel):
    """Detected visitor context from edge."""
    visitor_id: str
    device_type: Literal["mobile", "desktop", "tablet"]
    geo_country: str = "us"
    geo_region: Optional[str] = None
    user_agent: str
    ip_hash: str  # Hashed for privacy
    referrer: Optional[str] = None
    utm_source: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_medium: Optional[str] = None
    gclid: Optional[str] = None  # Google Click ID
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class GTMBinding(BaseModel):
    """Google Tag Manager container binding."""
    gtm_container_id: str  # e.g., "GTM-XXXXXX"
    page_container_id: str
    site_id: str
    session_id: str
    data_layer_endpoint: str  # WebSocket or SSE endpoint for real-time data
    bound_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class GTMEvent(BaseModel):
    """Real-time event from GTM data layer."""
    event_name: str
    session_id: str
    page_container_id: str
    site_id: str
    timestamp: str
    data: Dict[str, Any] = {}
    # Common GTM events
    page_view: Optional[bool] = None
    scroll_depth: Optional[int] = None
    click_element: Optional[str] = None
    form_submit: Optional[bool] = None
    video_progress: Optional[int] = None
    time_on_page: Optional[float] = None


class FunnelState(BaseModel):
    """Tracks user through multi-step funnel."""
    session_id: str
    site_id: str
    visitor_context: VisitorContext
    current_step: int = 1
    max_step_reached: int = 1
    total_steps: int = 4
    variant: Literal["a", "b"] = "a"
    containers: List[str] = []  # Container IDs for each step
    gtm_binding: Optional[GTMBinding] = None
    events: List[GTMEvent] = []
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = Field(default_factory=lambda: datetime.now().isoformat())
    converted: bool = False
    bounced: bool = False
    predicted_non_bounce: float = 0.7  # Initial assumption


class ComposeRequest(BaseModel):
    """Request to compose containers for a visitor."""
    site_id: str
    visitor_context: VisitorContext
    gtm_container_id: str = "GTM-DEFAULT"
    pre_warm_funnel: bool = True  # Assume non-bounce, warm all steps
    variant_override: Optional[Literal["a", "b"]] = None


class ComposeResponse(BaseModel):
    """Response from compose operation."""
    session_id: str
    site_id: str
    funnel_state: FunnelState
    landing_container_url: str
    gtm_binding: GTMBinding
    pre_warmed_steps: List[int] = []


class StepAdvanceRequest(BaseModel):
    """Request to advance user to next funnel step."""
    session_id: str
    site_id: str
    current_step: int
    gtm_events: List[GTMEvent] = []  # Events that triggered advance


class StepAdvanceResponse(BaseModel):
    """Response from step advance."""
    session_id: str
    next_step: int
    container_url: str
    funnel_complete: bool = False


class NeuralExport(BaseModel):
    """Export format for AI accelerator training."""
    site_id: str
    batch_id: str
    samples: List[Dict[str, Any]]
    context_features: List[str]
    action_space: List[str]
    reward_signals: List[str]
    metadata: Dict[str, Any] = {}
    export_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    format_version: str = "1.0"
    accelerator_hint: Literal["tpu", "gpu", "cpu"] = "gpu"
