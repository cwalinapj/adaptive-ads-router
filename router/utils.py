"""Utility functions for the Adaptive Ads Router."""

import os
import uuid
import json
import secrets
from datetime import datetime
from urllib.parse import urlencode, urlparse, parse_qsl, urlunparse
import redis.asyncio as redis


def get_config():
    return {
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
        "lam_regen_url": os.getenv("LAM_REGEN_URL", "http://lam-regen:8018/regenerate"),
        "first_100_threshold": int(os.getenv("FIRST_100_THRESHOLD", "100")),
        "neural_threshold": int(os.getenv("NEURAL_THRESHOLD", "1000")),
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.95")),
        "min_samples_per_arm": int(os.getenv("MIN_SAMPLES_PER_ARM", "3")),
        "admin_api_key": os.getenv("ADMIN_API_KEY"),
        "app_base_url": os.getenv("APP_BASE_URL", "http://localhost:8024"),
        "event_retention": int(os.getenv("EVENT_RETENTION", "5000")),
        "report_scheduler_enabled": os.getenv("REPORT_SCHEDULER_ENABLED", "true").lower() == "true",
        "report_scheduler_interval_seconds": int(os.getenv("REPORT_SCHEDULER_INTERVAL_SECONDS", "300")),
        "report_send_weekday": int(os.getenv("REPORT_SEND_WEEKDAY", "4")),
        "report_send_hour": int(os.getenv("REPORT_SEND_HOUR", "9")),
        "report_timezone": os.getenv("REPORT_TIMEZONE", "America/Los_Angeles"),
        "smtp_host": os.getenv("SMTP_HOST"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_username": os.getenv("SMTP_USERNAME"),
        "smtp_password": os.getenv("SMTP_PASSWORD"),
        "smtp_from": os.getenv("SMTP_FROM"),
        "smtp_use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
        "smtp_use_ssl": os.getenv("SMTP_USE_SSL", "false").lower() == "true",
    }


async def get_redis() -> redis.Redis:
    config = get_config()
    return await redis.from_url(config["redis_url"], decode_responses=True)


def generate_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:16]}"


def generate_tombstone_id() -> str:
    return f"tomb_{uuid.uuid4().hex[:12]}"


def generate_page_id(site_id: str, variant: str = None) -> str:
    variant = variant or uuid.uuid4().hex[:8]
    return f"{site_id}_page_{variant}"


def generate_owner_token() -> str:
    return secrets.token_urlsafe(24)


def now_iso() -> str:
    return datetime.now().isoformat()


def get_container_url(site_id: str, page_id: str) -> str:
    return f"http://pages.local/{site_id}/{page_id}"


def append_query_params(url: str, params: dict[str, str]) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({key: value for key, value in params.items() if value is not None})
    return urlunparse(parsed._replace(query=urlencode(query)))


def create_tombstone_record(
    site_id: str,
    terminated_page_id: str,
    successor_page_id: str,
    final_divergence_gap: float,
    total_lifetime_events: int,
    primary_failure_mode: str,
    llm_provider: str,
    hypothesis_was: str,
    actual_result: str
) -> dict:
    return {
        "tombstone_id": generate_tombstone_id(),
        "site_id": site_id,
        "terminated_page_id": terminated_page_id,
        "successor_page_id": successor_page_id,
        "final_divergence_gap": final_divergence_gap,
        "total_lifetime_events": total_lifetime_events,
        "primary_failure_mode": primary_failure_mode,
        "llm_provider": llm_provider,
        "hypothesis_was": hypothesis_was,
        "actual_result": actual_result,
        "created_at": now_iso()
    }


def log_event(event_type: str, data: dict):
    print(json.dumps({"timestamp": now_iso(), "event": event_type, **data}))
