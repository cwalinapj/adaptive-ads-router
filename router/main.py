"""
Adaptive Ads Router - Main FastAPI Application

Supports both standard Thompson Sampling and Contextual Bandits.
Contextual bandits learn optimal variants per device/geo/time segment.
"""

import os
import json
import csv
import secrets
import asyncio
import smtplib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from io import StringIO
from typing import Union, Optional, Literal
from zoneinfo import ZoneInfo
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
import redis.asyncio as redis
import httpx

from schemas import (
    RouteRequest, RouteResponse, OutcomeRequest, OutcomeResponse,
    SessionNeuralState, TombstoneRecord,
    DiffValidationRequest, DiffValidationResponse,
    SiteConfigRequest, SiteConfigResponse, SiteSummary, SiteVariant
)
from bandit import ThompsonSamplingBandit, ContextualBandit, Context
from diff_enforcer import RegimeDiffEnforcer, DiffEnforcer
from dashboard import render_dashboard, render_weekly_report
from utils import (
    get_config, generate_session_id, generate_page_id,
    get_container_url, now_iso, create_tombstone_record, log_event,
    append_query_params, generate_owner_token
)


# Type alias for either bandit type
BanditType = Union[ThompsonSamplingBandit, ContextualBandit]


# Redis client (global)
redis_client = None
report_scheduler_task = None


def site_config_key(site_id: str) -> str:
    return f"site_config:{site_id}"


def site_events_key(site_id: str) -> str:
    return f"events:{site_id}"


def delivery_log_key(site_id: str) -> str:
    return f"report_delivery:{site_id}"


def report_sent_week_key(site_id: str) -> str:
    return f"report_week_sent:{site_id}"


def report_attempted_week_key(site_id: str) -> str:
    return f"report_week_attempted:{site_id}"


def last_report_payload_key(site_id: str) -> str:
    return f"report_payload_last:{site_id}"


def build_default_site_config(site_id: str) -> dict:
    page_a = generate_page_id(site_id, "a")
    page_b = generate_page_id(site_id, "b")
    return {
        "site_id": site_id,
        "site_name": site_id.replace("-", " ").title(),
        "primary_goal": "lead",
        "report_email": None,
        "owner_token": generate_owner_token(),
        "variants": [
            {
                "page_id": page_a,
                "label": "Variant A",
                "url": get_container_url(site_id, page_a),
                "notes": "Default control variant"
            },
            {
                "page_id": page_b,
                "label": "Variant B",
                "url": get_container_url(site_id, page_b),
                "notes": "Default challenger variant"
            }
        ],
        "created_at": now_iso(),
        "updated_at": now_iso()
    }


async def get_site_config(site_id: str) -> dict:
    data = await redis_client.get(site_config_key(site_id))
    if data:
        return json.loads(data)

    config = build_default_site_config(site_id)
    await redis_client.set(site_config_key(site_id), json.dumps(config))
    return config


async def get_existing_site_config(site_id: str) -> Optional[dict]:
    data = await redis_client.get(site_config_key(site_id))
    return json.loads(data) if data else None


def extract_access_token(request: Request) -> Optional[str]:
    bearer = request.headers.get("Authorization", "")
    if bearer.lower().startswith("bearer "):
        return bearer.split(" ", 1)[1].strip()
    return (
        request.headers.get("X-AAR-Token")
        or request.query_params.get("token")
    )


async def list_owned_site_summaries(access_token: str) -> list[dict]:
    summaries = []
    async for key in redis_client.scan_iter(match="site_config:*"):
        config = json.loads(await redis_client.get(key))
        if secrets.compare_digest(config.get("owner_token", ""), access_token):
            summaries.append({
                "site_id": config["site_id"],
                "site_name": config["site_name"],
                "primary_goal": config["primary_goal"],
                "variant_count": len(config["variants"]),
                "created_at": config.get("created_at"),
                "updated_at": config["updated_at"]
            })
    return sorted(summaries, key=lambda item: item["site_id"])


def attach_management_fields(site_config: dict, request: Request, include_token: bool = False) -> dict:
    payload = dict(site_config)
    token = payload.get("owner_token")
    dashboard_url = None
    if token:
        dashboard_url = str(request.url_for("dashboard", site_id=site_config["site_id"])) + f"?token={token}"
    payload["dashboard_url"] = dashboard_url
    payload["management_token"] = token if include_token else None
    payload.pop("owner_token", None)
    return payload


def is_admin_token(token: Optional[str]) -> bool:
    admin_api_key = app.state.config.get("admin_api_key")
    return bool(admin_api_key and token and secrets.compare_digest(admin_api_key, token))


async def require_site_access(request: Request, site_id: str) -> tuple[dict, bool]:
    token = extract_access_token(request)
    existing = await get_existing_site_config(site_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Site not found")
    if is_admin_token(token):
        return existing, True
    owner_token = existing.get("owner_token")
    if token and owner_token and secrets.compare_digest(owner_token, token):
        return existing, False
    raise HTTPException(status_code=403, detail="Management token required")


async def save_site_config(site_id: str, request: SiteConfigRequest) -> dict:
    existing = await get_existing_site_config(site_id)
    created_at = existing.get("created_at") if existing else now_iso()
    owner_token = existing.get("owner_token") if existing else generate_owner_token()
    report_email = request.report_email if request.report_email is not None else (existing.get("report_email") if existing else None)

    variants = []
    for idx, variant in enumerate(request.variants):
        page_id = variant.page_id or generate_page_id(site_id, chr(97 + idx))
        variants.append({
            "page_id": page_id,
            "label": variant.label,
            "url": variant.url,
            "notes": variant.notes
        })

    payload = {
        "site_id": site_id,
        "site_name": request.site_name or site_id.replace("-", " ").title(),
        "primary_goal": request.primary_goal or "lead",
        "report_email": report_email,
        "owner_token": owner_token,
        "variants": variants,
        "created_at": created_at,
        "updated_at": now_iso()
    }
    await redis_client.set(site_config_key(site_id), json.dumps(payload))
    return payload


async def list_site_summaries() -> list[dict]:
    site_ids = set()
    for pattern in ("site_config:*", "bandit:*"):
        async for key in redis_client.scan_iter(match=pattern):
            site_ids.add(key.split(":", 1)[1])

    summaries = []
    for site_id in sorted(site_ids):
        config = await get_site_config(site_id)
        summaries.append({
            "site_id": site_id,
            "site_name": config["site_name"],
            "primary_goal": config["primary_goal"],
            "variant_count": len(config["variants"]),
            "created_at": config.get("created_at"),
            "updated_at": config["updated_at"]
        })
    return summaries


async def get_assignment(site_id: str, session_id: str) -> dict:
    raw = await redis_client.get(f"assignment:{site_id}:{session_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Session assignment not found")
    return json.loads(raw)


async def store_site_event(site_id: str, payload: dict) -> None:
    event = {"timestamp": now_iso(), "site_id": site_id, **payload}
    retention = app.state.config["event_retention"]
    key = site_events_key(site_id)
    await redis_client.lpush(key, json.dumps(event))
    await redis_client.ltrim(key, 0, retention - 1)


async def get_site_events(site_id: str, limit: int = 100) -> list[dict]:
    rows = await redis_client.lrange(site_events_key(site_id), 0, max(limit - 1, 0))
    return [json.loads(row) for row in rows]


def parse_date_value(value: Optional[str], field_name: str) -> Optional[datetime.date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}. Use YYYY-MM-DD.") from exc


def event_in_range(event: dict, start_date, end_date, event_type: Optional[str]) -> bool:
    if event_type and event.get("event_type") != event_type:
        return False
    timestamp = event.get("timestamp")
    if not timestamp:
        return False
    try:
        event_date = datetime.fromisoformat(timestamp).date()
    except ValueError:
        return False
    if start_date and event_date < start_date:
        return False
    if end_date and event_date > end_date:
        return False
    return True


async def filter_site_events(
    site_id: str,
    limit: int = 100,
    start: Optional[str] = None,
    end: Optional[str] = None,
    event_type: Optional[str] = None
) -> list[dict]:
    start_date = parse_date_value(start, "start")
    end_date = parse_date_value(end, "end")
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start must be on or before end")
    rows = await get_site_events(site_id, limit=app.state.config["event_retention"])
    filtered = [
        event for event in rows
        if event_in_range(event, start_date, end_date, event_type)
    ]
    return filtered[:limit]


def build_daily_report(events: list[dict]) -> list[dict]:
    daily = {}
    for event in events:
        day = event["timestamp"][:10]
        bucket = daily.setdefault(day, {
            "date": day,
            "routes": 0,
            "outcomes": 0,
            "conversions": 0,
            "revenue": 0.0,
        })
        if event.get("event_type") == "route":
            bucket["routes"] += 1
        elif event.get("event_type") == "outcome":
            bucket["outcomes"] += 1
            if event.get("converted"):
                bucket["conversions"] += 1
            bucket["revenue"] += float(event.get("revenue") or 0.0)
    results = []
    for day in sorted(daily.keys(), reverse=True):
        bucket = daily[day]
        routes = bucket["routes"]
        conversions = bucket["conversions"]
        bucket["conversion_rate"] = f"{((conversions / routes) * 100) if routes else 0:.2f}%"
        results.append(bucket)
    return results


def summarize_variant_performance(site_config: dict, events: list[dict]) -> list[dict]:
    variants = {
        variant["page_id"]: {
            "page_id": variant["page_id"],
            "label": variant["label"],
            "url": variant["url"],
            "routes": 0,
            "outcomes": 0,
            "conversions": 0,
            "revenue": 0.0,
        }
        for variant in site_config["variants"]
    }
    for event in events:
        page_id = event.get("page_id")
        if page_id not in variants:
            continue
        bucket = variants[page_id]
        if event.get("event_type") == "route":
            bucket["routes"] += 1
        elif event.get("event_type") == "outcome":
            bucket["outcomes"] += 1
            if event.get("converted"):
                bucket["conversions"] += 1
            bucket["revenue"] += float(event.get("revenue") or 0.0)
    results = []
    for bucket in variants.values():
        routes = bucket["routes"]
        bucket["conversion_rate"] = f"{((bucket['conversions'] / routes) * 100) if routes else 0:.2f}%"
        results.append(bucket)
    return sorted(results, key=lambda item: (item["conversions"], item["routes"], item["revenue"]), reverse=True)


def normalize_outcome_events(site_config: dict, events: list[dict], converted: bool) -> list[dict]:
    labels = {variant["page_id"]: variant["label"] for variant in site_config["variants"]}
    items = []
    for event in events:
        if event.get("event_type") != "outcome" or bool(event.get("converted")) != converted:
            continue
        items.append({
            "timestamp": event["timestamp"],
            "page_id": event.get("page_id"),
            "page_label": labels.get(event.get("page_id"), event.get("page_id")),
            "session_id": event.get("session_id"),
            "revenue": float(event.get("revenue") or 0.0),
        })
    return items[:5]


async def build_weekly_summary(
    site_id: str,
    site_config: dict,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> dict:
    if start is None and end is None:
        end = datetime.now().date().isoformat()
        start = (datetime.now().date() - timedelta(days=6)).isoformat()
    events = await filter_site_events(
        site_id,
        limit=app.state.config["event_retention"],
        start=start,
        end=end,
    )
    daily = build_daily_report(events)
    variants = summarize_variant_performance(site_config, events)
    summary = {
        "routes": sum(item["routes"] for item in daily),
        "outcomes": sum(item["outcomes"] for item in daily),
        "conversions": sum(item["conversions"] for item in daily),
        "revenue": round(sum(item["revenue"] for item in daily), 2),
    }
    summary["conversion_rate"] = f"{((summary['conversions'] / summary['routes']) * 100) if summary['routes'] else 0:.2f}%"
    summary["top_variant"] = variants[0] if variants and variants[0]["conversions"] > 0 else None
    return {
        "site_id": site_id,
        "site_name": site_config["site_name"],
        "filters": {"start": start, "end": end},
        "summary": summary,
        "variants": variants,
        "daily": daily,
        "recent_wins": normalize_outcome_events(site_config, events, converted=True),
        "recent_losses": normalize_outcome_events(site_config, events, converted=False),
    }


async def append_delivery_log(site_id: str, payload: dict) -> None:
    event = {"timestamp": now_iso(), "site_id": site_id, **payload}
    key = delivery_log_key(site_id)
    await redis_client.lpush(key, json.dumps(event))
    await redis_client.ltrim(key, 0, 99)


async def get_delivery_logs(site_id: str, limit: int = 20) -> list[dict]:
    rows = await redis_client.lrange(delivery_log_key(site_id), 0, max(0, limit - 1))
    return [json.loads(row) for row in rows]


async def save_last_report_payload(site_id: str, payload: dict) -> None:
    await redis_client.set(last_report_payload_key(site_id), json.dumps(payload))


async def get_last_report_payload(site_id: str) -> Optional[dict]:
    raw = await redis_client.get(last_report_payload_key(site_id))
    return json.loads(raw) if raw else None


def send_report_email(config: dict, to_email: str, subject: str, html_body: str, text_body: str) -> None:
    smtp_from = config.get("smtp_from")
    smtp_host = config.get("smtp_host")
    if not smtp_host or not smtp_from:
        raise RuntimeError("SMTP_HOST and SMTP_FROM must be configured")

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = smtp_from
    message["To"] = to_email
    message.attach(MIMEText(text_body, "plain"))
    message.attach(MIMEText(html_body, "html"))

    smtp_port = config.get("smtp_port", 587)
    smtp_username = config.get("smtp_username")
    smtp_password = config.get("smtp_password")
    use_ssl = config.get("smtp_use_ssl", False)
    use_tls = config.get("smtp_use_tls", True)

    if use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as server:
            if smtp_username:
                server.login(smtp_username, smtp_password or "")
            server.sendmail(smtp_from, [to_email], message.as_string())
        return

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
        if use_tls:
            server.starttls()
        if smtp_username:
            server.login(smtp_username, smtp_password or "")
        server.sendmail(smtp_from, [to_email], message.as_string())


def report_window_is_open(config: dict) -> tuple[bool, str]:
    now_local = datetime.now(ZoneInfo(config["report_timezone"]))
    week_id = now_local.strftime("%G-W%V")
    return (
        now_local.weekday() == config["report_send_weekday"] and now_local.hour == config["report_send_hour"],
        week_id,
    )


async def deliver_weekly_report(site_id: str, site_config: dict) -> None:
    await send_weekly_report(
        site_id=site_id,
        site_config=site_config,
        report_email=site_config.get("report_email"),
        enforce_schedule_window=True,
        skip_if_attempted=True,
    )


async def send_weekly_report(
    site_id: str,
    site_config: dict,
    report_email: Optional[str],
    enforce_schedule_window: bool,
    skip_if_attempted: bool,
) -> dict:
    if not report_email:
        return {"status": "skipped", "reason": "report_email is not configured"}
    config = app.state.config
    window_open, week_id = report_window_is_open(config)
    if enforce_schedule_window and not window_open:
        return {"status": "skipped", "reason": "outside delivery window", "week_id": week_id}

    last_sent_week = await redis_client.get(report_sent_week_key(site_id))
    last_attempt_week = await redis_client.get(report_attempted_week_key(site_id))
    if skip_if_attempted and (last_sent_week == week_id or last_attempt_week == week_id):
        return {"status": "skipped", "reason": "already attempted this week", "week_id": week_id}

    token = site_config.get("owner_token", "")
    report = await build_weekly_summary(site_id, site_config)
    base_url = config["app_base_url"].rstrip("/")
    report_url = f"{base_url}/reports/{site_id}/weekly-summary?token={token}"
    dashboard_url = f"{base_url}/dashboard/{site_id}?token={token}"
    html_body = render_weekly_report(site_config, report, report_url, dashboard_url)
    text_body = (
        f"Weekly report for {site_config['site_name']}\n"
        f"Routes: {report['summary']['routes']}\n"
        f"Conversions: {report['summary']['conversions']}\n"
        f"CVR: {report['summary']['conversion_rate']}\n"
        f"Revenue: {report['summary']['revenue']:.2f}\n"
        f"Dashboard: {dashboard_url}\n"
        f"JSON Report: {report_url}\n"
    )
    subject = f"[Adaptive Ads Router] Weekly report - {site_config['site_name']}"
    payload_snapshot = {
        "site_id": site_id,
        "week_id": week_id,
        "report_email": report_email,
        "subject": subject,
        "html_body": html_body,
        "text_body": text_body,
        "generated_at": now_iso(),
    }
    await save_last_report_payload(site_id, payload_snapshot)

    try:
        await asyncio.to_thread(send_report_email, config, report_email, subject, html_body, text_body)
        await redis_client.set(report_sent_week_key(site_id), week_id)
        await redis_client.set(report_attempted_week_key(site_id), week_id)
        await append_delivery_log(site_id, {
            "status": "sent",
            "week_id": week_id,
            "report_email": report_email,
            "summary": report["summary"],
        })
        return {"status": "sent", "week_id": week_id, "report_email": report_email}
    except Exception as exc:
        await redis_client.set(report_attempted_week_key(site_id), week_id)
        await append_delivery_log(site_id, {
            "status": "failed",
            "week_id": week_id,
            "report_email": report_email,
            "error": str(exc),
        })
        return {"status": "failed", "week_id": week_id, "report_email": report_email, "error": str(exc)}


async def resend_last_report_payload(site_id: str) -> dict:
    config = app.state.config
    payload = await get_last_report_payload(site_id)
    if not payload:
        return {"status": "failed", "error": "No previously generated report payload found"}

    report_email = payload.get("report_email")
    subject = payload.get("subject")
    html_body = payload.get("html_body")
    text_body = payload.get("text_body")
    week_id = payload.get("week_id")
    if not report_email or not subject or not html_body or not text_body:
        return {"status": "failed", "error": "Stored report payload is incomplete"}

    try:
        await asyncio.to_thread(send_report_email, config, report_email, subject, html_body, text_body)
        await append_delivery_log(site_id, {
            "status": "sent",
            "mode": "resend",
            "week_id": week_id,
            "report_email": report_email,
            "generated_at": payload.get("generated_at"),
        })
        return {
            "status": "sent",
            "mode": "resend",
            "week_id": week_id,
            "report_email": report_email,
            "generated_at": payload.get("generated_at"),
        }
    except Exception as exc:
        await append_delivery_log(site_id, {
            "status": "failed",
            "mode": "resend",
            "week_id": week_id,
            "report_email": report_email,
            "generated_at": payload.get("generated_at"),
            "error": str(exc),
        })
        return {
            "status": "failed",
            "mode": "resend",
            "week_id": week_id,
            "report_email": report_email,
            "generated_at": payload.get("generated_at"),
            "error": str(exc),
        }


async def report_scheduler_loop() -> None:
    while True:
        try:
            async for key in redis_client.scan_iter(match="site_config:*"):
                raw = await redis_client.get(key)
                if not raw:
                    continue
                site_config = json.loads(raw)
                await deliver_weekly_report(site_config["site_id"], site_config)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_event("report_scheduler_error", {"error": str(exc)})
        await asyncio.sleep(max(30, app.state.config["report_scheduler_interval_seconds"]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, report_scheduler_task
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    app.state.redis = redis_client
    app.state.config = get_config()
    print(f"Router started - Redis: {redis_url}")
    if app.state.config["report_scheduler_enabled"]:
        report_scheduler_task = asyncio.create_task(report_scheduler_loop())
    yield
    if report_scheduler_task is not None:
        report_scheduler_task.cancel()
        try:
            await report_scheduler_task
        except asyncio.CancelledError:
            pass
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
    site_config = await get_site_config(site_id)
    page_ids = [variant["page_id"] for variant in site_config["variants"]]
    key = f"bandit:{site_id}"
    data = await redis_client.get(key)

    if data:
        parsed = json.loads(data)
        if parsed.get("type") == "contextual":
            bandit = ContextualBandit.from_dict(parsed)
        else:
            bandit = ThompsonSamplingBandit.from_dict(parsed)
        bandit.sync_arms(page_ids)
        await save_bandit(bandit)
        return bandit

    if use_contextual_bandit():
        bandit = ContextualBandit(
            site_id=site_id,
            first_100_threshold=config["first_100_threshold"],
            neural_threshold=config["neural_threshold"],
            confidence_threshold=config["confidence_threshold"]
        )
        bandit_type = "contextual"
    else:
        bandit = ThompsonSamplingBandit(
            site_id=site_id,
            first_100_threshold=config["first_100_threshold"],
            neural_threshold=config["neural_threshold"],
            confidence_threshold=config["confidence_threshold"]
        )
        bandit_type = "standard"

    bandit.sync_arms(page_ids)
    await redis_client.set(key, bandit.to_json())
    log_event("bandit_created", {"site_id": site_id, "type": bandit_type})
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


async def apply_outcome_update(request: OutcomeRequest, background_tasks: Optional[BackgroundTasks] = None) -> OutcomeResponse:
    bandit = await get_or_create_bandit(request.site_id)

    context = None
    assignment_data = await redis_client.get(f"assignment:{request.site_id}:{request.session_id}")
    if assignment_data:
        assignment = json.loads(assignment_data)
        if assignment.get("context"):
            context = Context.from_dict(assignment["context"])

    try:
        if isinstance(bandit, ContextualBandit):
            bandit.update(request.page_id, request.converted, context)
            winner = bandit.get_winner(context)
            loser = bandit.get_loser(context) if winner else None
        else:
            bandit.update(request.page_id, request.converted)
            winner = bandit.get_winner()
            loser = bandit.get_loser() if winner else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await save_bandit(bandit)

    response = OutcomeResponse(
        recorded=True,
        regime=bandit.regime,
        winner_declared=winner is not None,
        winner_page_id=winner[0] if winner else None,
        should_regenerate=loser is not None,
        loser_page_id=loser
    )

    if loser and winner:
        if background_tasks is not None:
            background_tasks.add_task(create_tombstone_async, bandit, loser, winner[0], context)
        else:
            await create_tombstone_async(bandit, loser, winner[0], context)

    await store_site_event(request.site_id, {
        "event_type": "outcome",
        "page_id": request.page_id,
        "session_id": request.session_id,
        "converted": request.converted,
        "revenue": request.revenue,
        "winner_declared": response.winner_declared,
        "winner_page_id": response.winner_page_id,
        "regime": bandit.regime,
        "context": context.to_bucket() if context else None
    })

    log_event("outcome_recorded", {
        "site_id": request.site_id,
        "page_id": request.page_id,
        "converted": request.converted,
        "winner_declared": response.winner_declared,
        "context": context.to_bucket() if context else None
    })

    return response


# =============================================================================
# ROUTING ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "adaptive-ads-router"}


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Adaptive Ads Router</title>
    <style>
      :root {{
        --bg: #f7f5ef;
        --card: #fffdf8;
        --line: #ddd6c8;
        --ink: #16202a;
        --muted: #5a6270;
        --accent: #0f766e;
      }}
      * {{ box-sizing: border-box; }}
      body {{ font-family: sans-serif; margin: 40px auto; max-width: 1100px; padding: 0 20px; background: var(--bg); color: var(--ink); }}
      section {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 24px; }}
      .grid {{ display: grid; grid-template-columns: 1.25fr 0.95fr; gap: 18px; }}
      code, pre {{ font-family: monospace; }}
      pre {{ background: #f3eee4; padding: 16px; border-radius: 12px; overflow: auto; }}
      a {{ color: var(--accent); text-decoration: none; }}
      form {{ display: grid; gap: 12px; margin-top: 18px; }}
      .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
      label {{ display: grid; gap: 6px; font-size: 14px; color: var(--muted); }}
      input {{ width: 100%; padding: 12px 14px; border-radius: 10px; border: 1px solid var(--line); background: #fff; color: var(--ink); }}
      button {{ background: var(--accent); color: #fff; border: 0; border-radius: 10px; padding: 12px 16px; font-weight: 700; cursor: pointer; }}
      .muted {{ color: var(--muted); }}
      .status {{ min-height: 24px; color: var(--muted); }}
      @media (max-width: 900px) {{
        .grid, .two-col {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <div class="grid">
      <section>
        <h1>Adaptive Ads Router MVP</h1>
        <p>Configure landing page variants, route paid clicks, and record conversions from one service.</p>
        <p class="muted">Use this form to create your first site, then send ad traffic to <code>/r/&lt;site_id&gt;</code>.</p>
        <p class="muted">A management token is generated for each site. The dashboard link returned after setup is the private management URL.</p>
        <form id="create-site-form">
          <div class="two-col">
            <label>
              Site ID
              <input id="site-id" name="site_id" placeholder="acme-demo" required />
            </label>
            <label>
              Site name
              <input id="site-name" name="site_name" placeholder="Acme Demo" required />
            </label>
          </div>
          <label>
            Primary goal
            <input id="primary-goal" name="primary_goal" value="lead" />
          </label>
          <label>
            Client report email (optional)
            <input id="report-email" name="report_email" type="email" placeholder="client@example.com" />
          </label>
          <div class="two-col">
            <label>
              Variant A label
              <input id="variant-a-label" name="variant_a_label" value="Control" required />
            </label>
            <label>
              Variant A URL
              <input id="variant-a-url" name="variant_a_url" placeholder="https://example.com/landing-a" required />
            </label>
          </div>
          <div class="two-col">
            <label>
              Variant B label
              <input id="variant-b-label" name="variant_b_label" value="Challenger" required />
            </label>
            <label>
              Variant B URL
              <input id="variant-b-url" name="variant_b_url" placeholder="https://example.com/landing-b" required />
            </label>
          </div>
          <button type="submit">Create site and open dashboard</button>
          <div class="status" id="form-status"></div>
        </form>
      </section>
      <section>
        <h2>Private by Default</h2>
        <p class="muted">Configured sites, dashboards, and site settings require the site owner token or a global admin API key.</p>
        <h2>API example</h2>
        <pre>curl -X POST http://localhost:8024/sites/acme-demo \\
  -H 'content-type: application/json' \\
  -d '{{
    "site_name": "Acme Demo",
    "primary_goal": "lead",
    "report_email": "client@example.com",
    "variants": [
      {{"label": "Control", "url": "https://example.com/landing-a"}},
      {{"label": "Challenger", "url": "https://example.com/landing-b"}}
    ]
  }}'</pre>
      </section>
    </div>
    <script>
      const form = document.getElementById("create-site-form");
      const status = document.getElementById("form-status");

      form.addEventListener("submit", async (event) => {{
        event.preventDefault();
        status.textContent = "Creating site...";

        const payload = {{
          site_name: document.getElementById("site-name").value.trim(),
          primary_goal: document.getElementById("primary-goal").value.trim() || "lead",
          report_email: document.getElementById("report-email").value.trim() || null,
          variants: [
            {{
              label: document.getElementById("variant-a-label").value.trim(),
              url: document.getElementById("variant-a-url").value.trim(),
            }},
            {{
              label: document.getElementById("variant-b-label").value.trim(),
              url: document.getElementById("variant-b-url").value.trim(),
            }},
          ],
        }};

        const siteId = document.getElementById("site-id").value.trim();

        try {{
          const response = await fetch(`/sites/${{siteId}}`, {{
            method: "POST",
            headers: {{ "content-type": "application/json" }},
            body: JSON.stringify(payload),
          }});

          if (!response.ok) {{
            const data = await response.json().catch(() => ({{ detail: "Failed to create site." }}));
            throw new Error(data.detail || "Failed to create site.");
          }}

          const data = await response.json();
          if (!data.dashboard_url) {{
            throw new Error("Site created, but no dashboard URL was returned.");
          }}
          window.location.href = data.dashboard_url;
        }} catch (error) {{
          status.textContent = error.message;
        }}
      }});
    </script>
  </body>
</html>"""


@app.get("/sites", response_model=list[SiteSummary])
async def list_sites(request: Request):
    token = extract_access_token(request)
    if is_admin_token(token):
        return await list_site_summaries()
    if token:
        owned = await list_owned_site_summaries(token)
        if owned:
            return owned
    raise HTTPException(status_code=403, detail="Management token required")


@app.post("/sites/{site_id}", response_model=SiteConfigResponse)
async def configure_site(site_id: str, request: SiteConfigRequest, http_request: Request):
    existing = await get_existing_site_config(site_id)
    if existing is not None:
        await require_site_access(http_request, site_id)
    config = await save_site_config(site_id, request)
    bandit = await get_or_create_bandit(site_id)
    bandit.sync_arms([variant["page_id"] for variant in config["variants"]])
    await save_bandit(bandit)
    return attach_management_fields(config, http_request, include_token=True)


@app.get("/sites/{site_id}", response_model=SiteConfigResponse)
async def get_site(site_id: str, request: Request):
    config, _ = await require_site_access(request, site_id)
    return attach_management_fields(config, request)


@app.get("/dashboard/{site_id}", response_class=HTMLResponse)
async def dashboard(site_id: str, request: Request):
    await require_site_access(request, site_id)
    return HTMLResponse(render_dashboard(site_id))


async def make_route_decision(
    site_id: str,
    request: RouteRequest,
    http_request: Optional[Request] = None
) -> RouteResponse:
    """Shared routing logic for API and redirect flows."""
    existing = await get_existing_site_config(site_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Site not found")
    bandit = await get_or_create_bandit(site_id)
    site_config = existing
    context = extract_context(request, http_request) if isinstance(bandit, ContextualBandit) else None

    if isinstance(bandit, ContextualBandit):
        arm_index, page_id = bandit.select_arm(context)
    else:
        arm_index, page_id = bandit.select_arm()

    session_id = generate_session_id()
    variant = next((item for item in site_config["variants"] if item["page_id"] == page_id), None)
    assignment_data = {
        "page_id": page_id,
        "regime": bandit.regime,
        "ts": now_iso(),
        "visitor_id": request.visitor_id,
        "context": context.to_dict() if context else None
    }
    await redis_client.setex(
        f"assignment:{site_id}:{session_id}",
        3600,
        json.dumps(assignment_data)
    )

    await store_site_event(site_id, {
        "event_type": "route",
        "page_id": page_id,
        "session_id": session_id,
        "visitor_id": request.visitor_id,
        "destination_url": variant["url"] if variant else get_container_url(site_id, page_id),
        "regime": bandit.regime,
        "context": context.to_bucket() if context else None
    })

    log_event("route_decision", {
        "site_id": site_id,
        "page_id": page_id,
        "regime": bandit.regime,
        "visitor_id": request.visitor_id,
        "context": context.to_bucket() if context else None,
        "contextual": isinstance(bandit, ContextualBandit)
    })

    return RouteResponse(
        site_id=site_id,
        page_id=page_id,
        container_url=variant["url"] if variant else get_container_url(site_id, page_id),
        regime=bandit.regime,
        arm_index=arm_index,
        session_id=session_id
    )


@app.post("/route/{site_id}", response_model=RouteResponse)
async def route_traffic(site_id: str, request: RouteRequest, http_request: Request):
    """Route visitor to a page variant using Thompson Sampling (contextual if enabled)."""
    return await make_route_decision(site_id, request, http_request)


@app.get("/r/{site_id}")
async def route_and_redirect(
    site_id: str,
    visitor_id: Optional[str] = None,
    device_type: Literal["mobile", "desktop", "tablet"] = Query(default="desktop"),
    referrer: Optional[str] = None,
    utm_source: Optional[str] = None,
    utm_campaign: Optional[str] = None
):
    request = RouteRequest(
        visitor_id=visitor_id or generate_session_id(),
        device_type=device_type,
        referrer=referrer,
        utm_source=utm_source,
        utm_campaign=utm_campaign
    )
    response = await make_route_decision(site_id, request)
    site_config = await get_site_config(site_id)
    variant = next((item for item in site_config["variants"] if item["page_id"] == response.page_id), None)
    target_url = variant["url"] if variant else response.container_url
    redirect_url = append_query_params(
        target_url,
        {
            "aar_site_id": response.site_id,
            "aar_page_id": response.page_id,
            "aar_session_id": response.session_id,
        }
    )
    return RedirectResponse(url=redirect_url, status_code=307)


@app.get("/route/{site_id}")
async def route_traffic_simple(site_id: str, request: Request, device: str = "desktop", geo: str = "us"):
    """Simple GET routing for testing. Pass ?device=mobile&geo=uk for context."""
    existing = await get_existing_site_config(site_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Site not found")
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
    return await apply_outcome_update(request, background_tasks)


@app.get("/convert/{site_id}/{session_id}")
async def record_conversion(
    site_id: str,
    session_id: str,
    converted: bool = True,
    revenue: float = 0.0,
    redirect_to: Optional[str] = None
):
    assignment = await get_assignment(site_id, session_id)
    request = OutcomeRequest(
        site_id=site_id,
        page_id=assignment["page_id"],
        session_id=session_id,
        converted=converted,
        revenue=revenue
    )
    result = await apply_outcome_update(request)
    if redirect_to:
        return RedirectResponse(url=redirect_to, status_code=302)
    return {
        "recorded": result.recorded,
        "site_id": site_id,
        "page_id": assignment["page_id"],
        "session_id": session_id,
        "converted": converted
    }


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
async def get_stats(site_id: str, request: Request, device: str = None, geo: str = None):
    """
    Get current bandit stats for a site.
    For contextual bandits, optionally filter by device/geo.
    """
    await require_site_access(request, site_id)
    bandit = await get_or_create_bandit(site_id)
    context = None
    if isinstance(bandit, ContextualBandit) and (device or geo):
        context = Context.now(
            device=device or "desktop",
            geo=geo or "us"
        )

    stats = bandit.get_stats(context) if context else bandit.get_stats()
    site_config = await get_site_config(site_id)
    variants = {variant["page_id"]: variant for variant in site_config["variants"]}
    for arm in stats["arms"]:
        arm["label"] = variants.get(arm["page_id"], {}).get("label", arm["page_id"])
        arm["url"] = variants.get(arm["page_id"], {}).get("url", get_container_url(site_id, arm["page_id"]))
    return stats


@app.get("/events/{site_id}.csv")
async def export_events_csv(
    site_id: str,
    request: Request,
    limit: int = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    type: Optional[Literal["route", "outcome"]] = None,
):
    await require_site_access(request, site_id)
    events = await filter_site_events(
        site_id,
        limit=max(1, min(limit, 5000)),
        start=start,
        end=end,
        event_type=type
    )
    fieldnames = [
        "timestamp",
        "event_type",
        "site_id",
        "page_id",
        "session_id",
        "visitor_id",
        "destination_url",
        "converted",
        "revenue",
        "winner_declared",
        "winner_page_id",
        "regime",
        "context",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for event in events:
        row = {key: event.get(key) for key in fieldnames}
        if isinstance(row.get("context"), dict):
            row["context"] = json.dumps(row["context"], sort_keys=True)
        writer.writerow(row)
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{site_id}-events.csv"'
        }
    )


@app.get("/events/{site_id}")
async def get_events(
    site_id: str,
    request: Request,
    limit: int = 50,
    start: Optional[str] = None,
    end: Optional[str] = None,
    type: Optional[Literal["route", "outcome"]] = None,
):
    await require_site_access(request, site_id)
    events = await filter_site_events(
        site_id,
        limit=max(1, min(limit, 500)),
        start=start,
        end=end,
        event_type=type
    )
    return {
        "site_id": site_id,
        "count": len(events),
        "filters": {"start": start, "end": end, "type": type},
        "events": events
    }


@app.get("/reports/{site_id}/daily")
async def get_daily_report(
    site_id: str,
    request: Request,
    days: int = 7,
    start: Optional[str] = None,
    end: Optional[str] = None,
    type: Optional[Literal["route", "outcome"]] = None,
):
    await require_site_access(request, site_id)
    if start is None and end is None:
        days = max(1, min(days, 365))
        end = datetime.now().date().isoformat()
        start = (datetime.now().date() - timedelta(days=days - 1)).isoformat()
    events = await filter_site_events(
        site_id,
        limit=app.state.config["event_retention"],
        start=start,
        end=end,
        event_type=type
    )
    daily = build_daily_report(events)
    totals = {
        "routes": sum(item["routes"] for item in daily),
        "outcomes": sum(item["outcomes"] for item in daily),
        "conversions": sum(item["conversions"] for item in daily),
        "revenue": round(sum(item["revenue"] for item in daily), 2),
    }
    totals["conversion_rate"] = f"{((totals['conversions'] / totals['routes']) * 100) if totals['routes'] else 0:.2f}%"
    return {
        "site_id": site_id,
        "filters": {"start": start, "end": end, "type": type},
        "days": len(daily),
        "totals": totals,
        "daily": daily
    }


@app.get("/reports/{site_id}/weekly-summary")
async def get_weekly_summary(
    site_id: str,
    request: Request,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    site_config, _ = await require_site_access(request, site_id)
    return await build_weekly_summary(site_id, site_config, start=start, end=end)


@app.get("/reports/{site_id}/weekly-summary/html", response_class=HTMLResponse)
async def get_weekly_summary_html(
    site_id: str,
    request: Request,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    site_config, _ = await require_site_access(request, site_id)
    report = await build_weekly_summary(site_id, site_config, start=start, end=end)
    token = extract_access_token(request)
    origin = str(request.base_url).rstrip("/")
    dashboard_url = f"{origin}/dashboard/{site_id}"
    report_url = f"{origin}/reports/{site_id}/weekly-summary"
    if token:
        dashboard_url = f"{dashboard_url}?token={token}"
        report_url = f"{report_url}?token={token}"
    return HTMLResponse(render_weekly_report(site_config, report, report_url, dashboard_url))


@app.get("/reports/{site_id}/deliveries")
async def get_report_deliveries(site_id: str, request: Request, limit: int = 20):
    await require_site_access(request, site_id)
    deliveries = await get_delivery_logs(site_id, limit=max(1, min(limit, 100)))
    return {
        "site_id": site_id,
        "count": len(deliveries),
        "deliveries": deliveries
    }


@app.post("/reports/{site_id}/weekly-summary/send-test")
async def send_weekly_test_report(
    site_id: str,
    request: Request,
    email: Optional[str] = None,
):
    site_config, _ = await require_site_access(request, site_id)
    target_email = email or site_config.get("report_email")
    result = await send_weekly_report(
        site_id=site_id,
        site_config=site_config,
        report_email=target_email,
        enforce_schedule_window=False,
        skip_if_attempted=False,
    )
    return {
        "site_id": site_id,
        "result": result,
    }


@app.post("/reports/{site_id}/weekly-summary/resend-last")
async def resend_last_weekly_report(site_id: str, request: Request):
    await require_site_access(request, site_id)
    result = await resend_last_report_payload(site_id)
    status_code = 200 if result.get("status") == "sent" else 400
    return Response(
        content=json.dumps({"site_id": site_id, "result": result}),
        media_type="application/json",
        status_code=status_code,
    )


@app.get("/stats/{site_id}/contexts")
async def get_context_breakdown(site_id: str, request: Request):
    """
    Get per-context performance breakdown (contextual bandit only).
    Shows which variant wins for each device/geo/time segment.
    """
    await require_site_access(request, site_id)
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
async def validate_diff(request: DiffValidationRequest, http_request: Request):
    """Validate a proposed diff against regime policy."""
    await require_site_access(http_request, request.site_id)
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
async def get_allowed_changes(site_id: str, request: Request):
    """Get allowed change types for current regime."""
    await require_site_access(request, site_id)
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
async def get_tombstones(site_id: str, request: Request, limit: int = 10):
    """Get tombstone records for a site."""
    await require_site_access(request, site_id)
    tombstones = await redis_client.lrange(f"tombstones:{site_id}", -limit, -1)
    return {
        "site_id": site_id,
        "count": len(tombstones),
        "tombstones": [json.loads(t) for t in tombstones]
    }


@app.get("/neural-data/{site_id}")
async def get_neural_data(site_id: str, request: Request, limit: int = 100):
    """Get neural session data for training export."""
    await require_site_access(request, site_id)
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
async def reset_bandit(site_id: str, request: Request):
    """Reset bandit for a site (for testing)."""
    await require_site_access(request, site_id)
    await redis_client.delete(f"bandit:{site_id}")
    return {"reset": True, "site_id": site_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8024)
