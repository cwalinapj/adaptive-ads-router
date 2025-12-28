"""
Adaptive Ads Router - Main FastAPI Application

Supports both standard Thompson Sampling and Contextual Bandits.
Contextual bandits learn optimal variants per device/geo/time segment.
"""

import os
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Union

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
import httpx

from schemas import (
    RouteRequest, RouteResponse, OutcomeRequest, OutcomeResponse,
    SessionNeuralState, TombstoneRecord,
    DiffValidationRequest, DiffValidationResponse
)
from bandit import ThompsonSamplingBandit, ContextualBandit, Context
from diff_enforcer import RegimeDiffEnforcer, DiffEnforcer
from utils import (
    get_config, generate_session_id, generate_page_id,
    get_container_url, now_iso, create_tombstone_record, log_event
)


# Type alias for either bandit type
BanditType = Union[ThompsonSamplingBandit, ContextualBandit]


# Redis client (global)
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    app.state.redis = redis_client
    app.state.config = get_config()
    print(f"Router started - Redis: {redis_url}")
    yield
    await redis_client.close()


app = FastAPI(
    title="Adaptive Ads Router",
    description="High-velocity bandit system for Google Ads landing pages",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SESSION INGESTION
# =============================================================================

@app.post("/session")
async def ingest_session(state: SessionNeuralState):
    """Ingest session neural state for LAM training."""
    key = f"session:{state.session_id}"
    await redis_client.setex(key, 86400, state.model_dump_json())
    
    # Also append to site's neural data list
    neural_key = f"neural:{state.site_id}"
    await redis_client.rpush(neural_key, state.model_dump_json())
    await redis_client.ltrim(neural_key, -10000, -1)  # Keep last 10k
    
    return {"stored": True, "session_id": state.session_id}


# =============================================================================
# BANDIT HELPERS
# =============================================================================

def use_contextual_bandit() -> bool:
    """Check if contextual bandit is enabled."""
    return os.getenv("USE_CONTEXTUAL_BANDIT", "true").lower() == "true"


async def get_or_create_bandit(site_id: str) -> BanditType:
    """Get or create bandit for a site. Supports both standard and contextual."""
    config = app.state.config
    key = f"bandit:{site_id}"
    data = await redis_client.get(key)

    if data:
        parsed = json.loads(data)
        # Check if it's a contextual bandit
        if parsed.get("type") == "contextual":
            return ContextualBandit.from_dict(parsed)
        return ThompsonSamplingBandit.from_dict(parsed)

    # Create new bandit based on config
    if use_contextual_bandit():
        bandit = ContextualBandit(
            site_id=site_id,
            first_100_threshold=config["first_100_threshold"],
            neural_threshold=config["neural_threshold"],
            confidence_threshold=config["confidence_threshold"]
        )
        log_event("bandit_created", {"site_id": site_id, "type": "contextual"})
    else:
        bandit = ThompsonSamplingBandit(
            site_id=site_id,
            first_100_threshold=config["first_100_threshold"],
            neural_threshold=config["neural_threshold"],
            confidence_threshold=config["confidence_threshold"]
        )
        log_event("bandit_created", {"site_id": site_id, "type": "standard"})

    bandit.add_arm(generate_page_id(site_id, "a"))
    bandit.add_arm(generate_page_id(site_id, "b"))
    await redis_client.set(key, bandit.to_json())
    return bandit


async def save_bandit(bandit: BanditType):
    """Save bandit state to Redis."""
    await redis_client.set(f"bandit:{bandit.site_id}", bandit.to_json())


def extract_context(request: RouteRequest, http_request: Request = None) -> Context:
    """Extract visitor context from request data and headers."""
    # Get device type from request
    device = request.device_type

    # Try to get geo from headers (CloudFlare, AWS, etc.)
    geo = "us"  # default
    if http_request:
        geo = (
            http_request.headers.get("CF-IPCountry", "").lower() or
            http_request.headers.get("CloudFront-Viewer-Country", "").lower() or
            "us"
        )

    return Context.now(device=device, geo=geo)


# =============================================================================
# ROUTING ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "adaptive-ads-router"}


@app.post("/route/{site_id}", response_model=RouteResponse)
async def route_traffic(site_id: str, request: RouteRequest, http_request: Request):
    """Route visitor to a page variant using Thompson Sampling (contextual if enabled)."""
    bandit = await get_or_create_bandit(site_id)

    # Extract context for contextual bandit
    context = extract_context(request, http_request)

    # Select arm based on bandit type
    if isinstance(bandit, ContextualBandit):
        arm_index, page_id = bandit.select_arm(context)
    else:
        arm_index, page_id = bandit.select_arm()

    session_id = generate_session_id()

    # Store session assignment with context
    assignment_data = {
        "page_id": page_id,
        "regime": bandit.regime,
        "ts": now_iso(),
        "context": context.to_dict() if isinstance(bandit, ContextualBandit) else None
    }
    await redis_client.setex(
        f"assignment:{site_id}:{session_id}",
        3600,
        json.dumps(assignment_data)
    )

    log_event("route_decision", {
        "site_id": site_id,
        "page_id": page_id,
        "regime": bandit.regime,
        "visitor_id": request.visitor_id,
        "context": context.to_bucket() if isinstance(bandit, ContextualBandit) else None,
        "contextual": isinstance(bandit, ContextualBandit)
    })

    return RouteResponse(
        site_id=site_id,
        page_id=page_id,
        container_url=get_container_url(site_id, page_id),
        regime=bandit.regime,
        arm_index=arm_index,
        session_id=session_id
    )


@app.get("/route/{site_id}")
async def route_traffic_simple(site_id: str, request: Request, device: str = "desktop", geo: str = "us"):
    """Simple GET routing for testing. Pass ?device=mobile&geo=uk for context."""
    bandit = await get_or_create_bandit(site_id)
    session_id = generate_session_id()

    # Create context from query params
    context = Context.now(device=device, geo=geo)

    # Select arm based on bandit type
    if isinstance(bandit, ContextualBandit):
        arm_index, page_id = bandit.select_arm(context)
        context_bucket = context.to_bucket()
    else:
        arm_index, page_id = bandit.select_arm()
        context_bucket = None

    return {
        "site_id": site_id,
        "page_id": page_id,
        "regime": bandit.regime,
        "session_id": session_id,
        "context": context_bucket,
        "contextual": isinstance(bandit, ContextualBandit)
    }


# =============================================================================
# OUTCOME RECORDING
# =============================================================================

@app.post("/outcome", response_model=OutcomeResponse)
async def record_outcome(request: OutcomeRequest, background_tasks: BackgroundTasks):
    """Record conversion outcome and update bandit."""
    bandit = await get_or_create_bandit(request.site_id)

    # Try to retrieve context from session assignment
    context = None
    assignment_data = await redis_client.get(f"assignment:{request.site_id}:{request.session_id}")
    if assignment_data:
        assignment = json.loads(assignment_data)
        if assignment.get("context"):
            context = Context.from_dict(assignment["context"])

    # Update bandit with context if available
    if isinstance(bandit, ContextualBandit):
        bandit.update(request.page_id, request.converted, context)
        winner = bandit.get_winner(context)
        loser = bandit.get_loser(context) if winner else None
    else:
        bandit.update(request.page_id, request.converted)
        winner = bandit.get_winner()
        loser = bandit.get_loser() if winner else None

    await save_bandit(bandit)

    response = OutcomeResponse(
        recorded=True,
        regime=bandit.regime,
        winner_declared=winner is not None,
        winner_page_id=winner[0] if winner else None,
        should_regenerate=loser is not None,
        loser_page_id=loser
    )

    # Create tombstone if we have a loser
    if loser and winner:
        background_tasks.add_task(create_tombstone_async, bandit, loser, winner[0], context)

    log_event("outcome_recorded", {
        "site_id": request.site_id,
        "page_id": request.page_id,
        "converted": request.converted,
        "winner_declared": response.winner_declared,
        "context": context.to_bucket() if context else None
    })

    return response


async def create_tombstone_async(
    bandit: BanditType,
    loser_id: str,
    winner_id: str,
    context: Context = None
):
    """Create tombstone record for terminated page."""
    loser_arm = next((a for a in bandit.arms if a.page_id == loser_id), None)
    if not loser_arm:
        return
    
    tombstone = create_tombstone_record(
        site_id=bandit.site_id,
        terminated_page_id=loser_id,
        successor_page_id=winner_id,
        final_divergence_gap=bandit.get_divergence(),
        total_lifetime_events=loser_arm.total,
        primary_failure_mode="low_conversion",
        llm_provider="anthropic",
        hypothesis_was="Variant would improve conversions",
        actual_result=f"Rate: {loser_arm.mean:.2%}"
    )
    
    await redis_client.rpush(f"tombstones:{bandit.site_id}", json.dumps(tombstone))
    log_event("tombstone_created", tombstone)


# =============================================================================
# STATS & DIFF VALIDATION
# =============================================================================

@app.get("/stats/{site_id}")
async def get_stats(site_id: str, device: str = None, geo: str = None):
    """
    Get current bandit stats for a site.
    For contextual bandits, optionally filter by device/geo.
    """
    bandit = await get_or_create_bandit(site_id)

    # Build context if filters provided and bandit is contextual
    context = None
    if isinstance(bandit, ContextualBandit) and (device or geo):
        context = Context.now(
            device=device or "desktop",
            geo=geo or "us"
        )
        return bandit.get_stats(context)

    return bandit.get_stats()


@app.get("/stats/{site_id}/contexts")
async def get_context_breakdown(site_id: str):
    """
    Get per-context performance breakdown (contextual bandit only).
    Shows which variant wins for each device/geo/time segment.
    """
    bandit = await get_or_create_bandit(site_id)

    if not isinstance(bandit, ContextualBandit):
        return {
            "site_id": site_id,
            "error": "Not a contextual bandit",
            "contextual": False
        }

    return {
        "site_id": site_id,
        "contextual": True,
        "breakdown": bandit.get_context_breakdown(),
        "top_contexts": sorted(
            bandit.context_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
    }


@app.post("/validate-diff", response_model=DiffValidationResponse)
async def validate_diff(request: DiffValidationRequest):
    """Validate a proposed diff against regime policy."""
    bandit = await get_or_create_bandit(request.site_id)
    enforcer = RegimeDiffEnforcer(bandit.regime)
    
    allowed, violations, diff_score = enforcer.validate(
        request.proposed_diff.changes,
        request.proposed_diff.hypothesis
    )
    
    return DiffValidationResponse(
        allowed=allowed,
        regime=bandit.regime,
        diff_score=diff_score,
        max_allowed=enforcer.get_max_diff_score(),
        violations=violations,
        reason="; ".join(violations) if violations else None
    )


@app.get("/allowed-changes/{site_id}")
async def get_allowed_changes(site_id: str):
    """Get allowed change types for current regime."""
    bandit = await get_or_create_bandit(site_id)
    enforcer = RegimeDiffEnforcer(bandit.regime)
    
    return {
        "site_id": site_id,
        "regime": bandit.regime,
        "allowed": enforcer.get_allowed_changes(),
        "max_diff_score": enforcer.get_max_diff_score()
    }


# =============================================================================
# TOMBSTONES & NEURAL DATA
# =============================================================================

@app.get("/tombstones/{site_id}")
async def get_tombstones(site_id: str, limit: int = 10):
    """Get tombstone records for a site."""
    tombstones = await redis_client.lrange(f"tombstones:{site_id}", -limit, -1)
    return {
        "site_id": site_id,
        "count": len(tombstones),
        "tombstones": [json.loads(t) for t in tombstones]
    }


@app.get("/neural-data/{site_id}")
async def get_neural_data(site_id: str, limit: int = 100):
    """Get neural session data for training export."""
    data = await redis_client.lrange(f"neural:{site_id}", -limit, -1)
    return {
        "site_id": site_id,
        "count": len(data),
        "sessions": [json.loads(d) for d in data]
    }


# =============================================================================
# ADMIN
# =============================================================================

@app.post("/reset/{site_id}")
async def reset_bandit(site_id: str):
    """Reset bandit for a site (for testing)."""
    await redis_client.delete(f"bandit:{site_id}")
    return {"reset": True, "site_id": site_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8024)
