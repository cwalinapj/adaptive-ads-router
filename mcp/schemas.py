"""Schemas for Docker MCP Service."""

from pydantic import BaseModel
from typing import Any, Optional, Dict, List, Literal


class ContainerInfo(BaseModel):
    container_id: str
    site_id: str
    page_id: str
    status: Literal["running", "stopped", "creating", "error"]
    url: str
    created_at: str
    last_health_check: Optional[str] = None


class CreateContainerRequest(BaseModel):
    site_id: str
    page_id: str
    base_image: str = "nginx:alpine"
    html_content: Optional[str] = None
    config: Dict[str, Any] = {}


class CreateContainerResponse(BaseModel):
    container_id: str
    page_id: str
    url: str
    status: str


class DiffProposal(BaseModel):
    site_id: str
    source_page_id: str
    target_page_id: str
    changes: Dict[str, Any]
    hypothesis: str


class DiffValidationResult(BaseModel):
    allowed: bool
    diff_score: float
    max_allowed: float
    violations: List[str] = []
    reason: Optional[str] = None
