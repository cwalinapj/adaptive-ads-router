"""Utility functions for the Adaptive Ads Router."""

import os
import uuid
import json
from datetime import datetime
import redis.asyncio as redis


def get_config():
    return {
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
        "lam_regen_url": os.getenv("LAM_REGEN_URL", "http://lam-regen:8018/regenerate"),
        "first_100_threshold": int(os.getenv("FIRST_100_THRESHOLD", "100")),
        "neural_threshold": int(os.getenv("NEURAL_THRESHOLD", "1000")),
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.95")),
        "min_samples_per_arm": int(os.getenv("MIN_SAMPLES_PER_ARM", "3")),
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


def now_iso() -> str:
    return datetime.now().isoformat()


def get_container_url(site_id: str, page_id: str) -> str:
    return f"http://pages.local/{site_id}/{page_id}"


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
