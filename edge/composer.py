"""
Edge Composer Service
Handles visitor context detection, container composition, GTM binding, and funnel tracking.
"""

import os
import json
import asyncio
import subprocess
import uuid
import hashlib
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
import httpx

from schemas import (
    VisitorContext, GTMBinding, GTMEvent, FunnelState,
    ComposeRequest, ComposeResponse, StepAdvanceRequest, StepAdvanceResponse,
    NeuralExport
)


# =============================================================================
# APP SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await redis.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379"),
        decode_responses=True
    )
    # Active WebSocket connections for GTM real-time data
    app.state.gtm_connections = {}  # session_id -> WebSocket
    print("Edge Composer started")
    yield
    await app.state.redis.close()


app = FastAPI(title="Edge Composer", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def now_iso():
    return datetime.now().isoformat()


def generate_session_id():
    return f"edge_{uuid.uuid4().hex[:16]}"


# =============================================================================
# VISITOR CONTEXT DETECTION
# =============================================================================

def detect_device(user_agent: str) -> str:
    """Detect device type from user agent."""
    ua_lower = user_agent.lower()
    if "mobile" in ua_lower or "android" in ua_lower or "iphone" in ua_lower:
        return "mobile"
    if "tablet" in ua_lower or "ipad" in ua_lower:
        return "tablet"
    return "desktop"


def detect_geo(request: Request) -> tuple:
    """Detect geo from headers (CloudFlare, AWS, etc.) or IP."""
    # CloudFlare headers
    country = request.headers.get("CF-IPCountry", "").lower()
    if country:
        return country, request.headers.get("CF-Region", "")

    # AWS CloudFront
    country = request.headers.get("CloudFront-Viewer-Country", "").lower()
    if country:
        return country, ""

    # Fallback
    return "us", ""


def hash_ip(ip: str) -> str:
    """Hash IP for privacy."""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def extract_visitor_context(request: Request) -> VisitorContext:
    """Extract full visitor context from request."""
    user_agent = request.headers.get("User-Agent", "unknown")
    client_ip = request.headers.get("X-Forwarded-For", request.client.host).split(",")[0]
    geo_country, geo_region = detect_geo(request)

    return VisitorContext(
        visitor_id=request.query_params.get("visitor_id", generate_session_id()),
        device_type=detect_device(user_agent),
        geo_country=geo_country or "us",
        geo_region=geo_region,
        user_agent=user_agent,
        ip_hash=hash_ip(client_ip),
        referrer=request.headers.get("Referer"),
        utm_source=request.query_params.get("utm_source"),
        utm_campaign=request.query_params.get("utm_campaign"),
        utm_medium=request.query_params.get("utm_medium"),
        gclid=request.query_params.get("gclid"),
    )


# =============================================================================
# MAKEFILE EXECUTION
# =============================================================================

async def run_make(target: str, **kwargs) -> dict:
    """Run Makefile target with context variables."""
    env = os.environ.copy()
    for key, value in kwargs.items():
        env[key.upper()] = str(value)

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["make", "-C", "/app", target],
            capture_output=True,
            text=True,
            env=env,
            timeout=30
        )
        # Parse JSON output from Makefile
        for line in result.stdout.strip().split("\n"):
            if line.startswith("{"):
                return json.loads(line)
        return {"status": "completed", "stdout": result.stdout}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "Container compose timed out"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# GTM BINDING & REAL-TIME DATA
# =============================================================================

def create_gtm_binding(
    gtm_container_id: str,
    page_container_id: str,
    site_id: str,
    session_id: str
) -> GTMBinding:
    """Create GTM container binding for real-time data flow."""
    return GTMBinding(
        gtm_container_id=gtm_container_id,
        page_container_id=page_container_id,
        site_id=site_id,
        session_id=session_id,
        data_layer_endpoint=f"ws://edge:8040/gtm/{session_id}"
    )


@app.websocket("/gtm/{session_id}")
async def gtm_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time GTM data layer events.
    The page container sends events here, which are:
    1. Stored in Redis for the session
    2. Forwarded to the router for bandit updates
    3. Streamed to any listening AI accelerators
    """
    await websocket.accept()
    app.state.gtm_connections[session_id] = websocket

    try:
        while True:
            data = await websocket.receive_json()
            event = GTMEvent(
                event_name=data.get("event", "unknown"),
                session_id=session_id,
                page_container_id=data.get("container_id", ""),
                site_id=data.get("site_id", ""),
                timestamp=now_iso(),
                data=data.get("data", {}),
                scroll_depth=data.get("scroll_depth"),
                click_element=data.get("click_element"),
                form_submit=data.get("form_submit"),
                time_on_page=data.get("time_on_page"),
            )

            # Store event in Redis
            await app.state.redis.rpush(
                f"gtm_events:{session_id}",
                event.model_dump_json()
            )
            await app.state.redis.expire(f"gtm_events:{session_id}", 86400)

            # Update funnel state with event
            await process_gtm_event(event)

            # Forward to router for real-time bandit updates
            await forward_to_router(event)

    except WebSocketDisconnect:
        del app.state.gtm_connections[session_id]


async def forward_to_router(event: GTMEvent):
    """Forward GTM event to router for real-time processing."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://router:8024/session",
                json={
                    "session_id": event.session_id,
                    "site_id": event.site_id,
                    "page_id": event.page_container_id,
                    "llm_provider": "gtm",
                    "timestamp": event.timestamp,
                    "dwell_time": event.time_on_page or 0,
                    "max_scroll": (event.scroll_depth or 0) / 100,
                    "interaction_count": 1 if event.click_element else 0,
                    "cta_intent_score": 0.8 if event.form_submit else 0.0,
                    "conversion": event.form_submit or False,
                    "revenue_value": event.data.get("revenue", 0)
                },
                timeout=5.0
            )
    except Exception as e:
        print(f"[forward_to_router] Error: {e}")


async def process_gtm_event(event: GTMEvent):
    """Process GTM event and update funnel state."""
    funnel_key = f"funnel:{event.session_id}"
    funnel_data = await app.state.redis.get(funnel_key)

    if not funnel_data:
        return

    funnel = FunnelState(**json.loads(funnel_data))
    funnel.events.append(event)
    funnel.last_activity = now_iso()

    # Check for conversion event
    if event.form_submit or event.event_name in ["purchase", "lead", "signup"]:
        funnel.converted = True

    # Update bounce prediction based on engagement
    if event.scroll_depth and event.scroll_depth > 50:
        funnel.predicted_non_bounce = min(0.95, funnel.predicted_non_bounce + 0.1)
    if event.time_on_page and event.time_on_page > 30:
        funnel.predicted_non_bounce = min(0.95, funnel.predicted_non_bounce + 0.1)

    await app.state.redis.setex(funnel_key, 86400, funnel.model_dump_json())


# =============================================================================
# COMPOSE ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "edge-composer"}


@app.post("/compose", response_model=ComposeResponse)
async def compose_containers(req: ComposeRequest):
    """
    Main entry point: compose containers for incoming visitor.
    1. Detect context (already provided or from headers)
    2. Create funnel state
    3. Bind GTM container
    4. Trigger Makefile to spin up containers
    5. Pre-warm funnel steps (assuming non-bounce)
    """
    session_id = generate_session_id()
    ctx = req.visitor_context

    # Determine variant (from router or override)
    variant = req.variant_override
    if not variant:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://router:8024/route/{req.site_id}",
                    json={
                        "visitor_id": ctx.visitor_id,
                        "device_type": ctx.device_type,
                        "referrer": ctx.referrer,
                        "utm_source": ctx.utm_source,
                        "utm_campaign": ctx.utm_campaign
                    },
                    timeout=5.0
                )
                if resp.status_code == 200:
                    route_data = resp.json()
                    variant = "a" if "page_a" in route_data.get("page_id", "") else "b"
        except Exception:
            variant = "a"  # Default fallback

    # Compose landing page container via Makefile
    compose_result = await run_make(
        "compose",
        site_id=req.site_id,
        device=ctx.device_type,
        geo=ctx.geo_country,
        variant=variant
    )

    page_container_id = compose_result.get("container", f"page_{req.site_id}_{variant}_step1")

    # Create GTM binding
    gtm_binding = create_gtm_binding(
        gtm_container_id=req.gtm_container_id,
        page_container_id=page_container_id,
        site_id=req.site_id,
        session_id=session_id
    )

    # Create funnel state
    funnel = FunnelState(
        session_id=session_id,
        site_id=req.site_id,
        visitor_context=ctx,
        variant=variant,
        containers=[page_container_id],
        gtm_binding=gtm_binding,
    )

    # Store funnel state
    await app.state.redis.setex(
        f"funnel:{session_id}",
        86400,
        funnel.model_dump_json()
    )

    # Pre-warm funnel steps (assuming non-bounce)
    pre_warmed = []
    if req.pre_warm_funnel:
        warm_result = await run_make(
            "warm-funnel",
            site_id=req.site_id,
            device=ctx.device_type,
            geo=ctx.geo_country,
            variant=variant,
            funnel_depth=4
        )
        if warm_result.get("status") == "funnel_warmed":
            pre_warmed = list(range(1, 5))

    return ComposeResponse(
        session_id=session_id,
        site_id=req.site_id,
        funnel_state=funnel,
        landing_container_url=f"http://pages.local/{req.site_id}/{page_container_id}",
        gtm_binding=gtm_binding,
        pre_warmed_steps=pre_warmed
    )


@app.get("/compose/{site_id}")
async def compose_from_request(site_id: str, request: Request):
    """
    Simple GET endpoint that extracts context from request headers.
    Used for direct traffic from Google Ads.
    """
    ctx = extract_visitor_context(request)

    compose_req = ComposeRequest(
        site_id=site_id,
        visitor_context=ctx,
        gtm_container_id=request.query_params.get("gtm", "GTM-DEFAULT"),
        pre_warm_funnel=True
    )

    return await compose_containers(compose_req)


# =============================================================================
# FUNNEL NAVIGATION
# =============================================================================

@app.post("/advance", response_model=StepAdvanceResponse)
async def advance_funnel_step(req: StepAdvanceRequest):
    """Advance user to next funnel step."""
    funnel_key = f"funnel:{req.session_id}"
    funnel_data = await app.state.redis.get(funnel_key)

    if not funnel_data:
        raise HTTPException(404, "Funnel session not found")

    funnel = FunnelState(**json.loads(funnel_data))

    if req.current_step != funnel.current_step:
        raise HTTPException(400, f"Step mismatch: expected {funnel.current_step}")

    next_step = funnel.current_step + 1
    funnel_complete = next_step > funnel.total_steps

    if not funnel_complete:
        # Compose next step container if not pre-warmed
        compose_result = await run_make(
            f"step-{next_step}",
            site_id=funnel.site_id,
            device=funnel.visitor_context.device_type,
            geo=funnel.visitor_context.geo_country,
            variant=funnel.variant
        )

        container_id = compose_result.get("container", f"page_{funnel.site_id}_{funnel.variant}_step{next_step}")
        funnel.containers.append(container_id)
        funnel.current_step = next_step
        funnel.max_step_reached = max(funnel.max_step_reached, next_step)
    else:
        funnel.converted = True

    # Store GTM events that triggered advance
    funnel.events.extend(req.gtm_events)
    funnel.last_activity = now_iso()

    await app.state.redis.setex(funnel_key, 86400, funnel.model_dump_json())

    container_url = f"http://pages.local/{funnel.site_id}/{funnel.containers[-1]}" if not funnel_complete else ""

    return StepAdvanceResponse(
        session_id=req.session_id,
        next_step=next_step,
        container_url=container_url,
        funnel_complete=funnel_complete
    )


@app.get("/funnel/{session_id}")
async def get_funnel_state(session_id: str):
    """Get current funnel state."""
    funnel_data = await app.state.redis.get(f"funnel:{session_id}")
    if not funnel_data:
        raise HTTPException(404, "Funnel not found")
    return json.loads(funnel_data)


# =============================================================================
# NEURAL EXPORT FOR AI ACCELERATORS
# =============================================================================

@app.get("/export/{site_id}", response_model=NeuralExport)
async def export_for_accelerator(site_id: str, limit: int = 1000):
    """
    Export funnel data as JSON for AI accelerator training.
    Returns (context, action, reward) tuples.
    """
    # Get all funnel sessions for this site
    keys = await app.state.redis.keys(f"funnel:*")
    samples = []

    for key in keys[:limit]:
        funnel_data = await app.state.redis.get(key)
        if not funnel_data:
            continue

        funnel = FunnelState(**json.loads(funnel_data))
        if funnel.site_id != site_id:
            continue

        # Build training sample
        sample = {
            "context": {
                "device": funnel.visitor_context.device_type,
                "geo": funnel.visitor_context.geo_country,
                "utm_source": funnel.visitor_context.utm_source,
                "referrer": funnel.visitor_context.referrer,
                "hour_of_day": datetime.fromisoformat(funnel.started_at).hour,
            },
            "action": {
                "variant": funnel.variant,
                "funnel_steps": funnel.total_steps,
            },
            "reward": {
                "converted": funnel.converted,
                "bounced": funnel.bounced,
                "max_step": funnel.max_step_reached,
                "engagement_score": len(funnel.events) / 10,  # Normalized
            },
            "trajectory": [
                {
                    "step": i + 1,
                    "events": len([e for e in funnel.events if f"step{i+1}" in e.page_container_id])
                }
                for i in range(funnel.max_step_reached)
            ]
        }
        samples.append(sample)

    return NeuralExport(
        site_id=site_id,
        batch_id=f"batch_{uuid.uuid4().hex[:8]}",
        samples=samples,
        context_features=["device", "geo", "utm_source", "referrer", "hour_of_day"],
        action_space=["variant_a", "variant_b"],
        reward_signals=["converted", "bounced", "max_step", "engagement_score"],
        metadata={
            "total_samples": len(samples),
            "conversion_rate": sum(1 for s in samples if s["reward"]["converted"]) / max(len(samples), 1)
        },
        accelerator_hint="gpu"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8040)
