import asyncio
import json
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

sys.path.insert(0, "/Users/root1/adaptive-ads-router/router")
import main  # noqa: E402


class FakeRedis:
    def __init__(self):
        self.kv = {}
        self.lists = {}
        self.expiry = {}

    async def ping(self):
        return True

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        if ex is not None:
            self.expiry[key] = ex
        return True

    async def setex(self, key, ttl, value):
        self.kv[key] = value
        self.expiry[key] = ttl

    async def incr(self, key):
        current = int(self.kv.get(key, "0")) + 1
        self.kv[key] = str(current)
        return current

    async def expire(self, key, ttl):
        self.expiry[key] = ttl
        return True

    async def delete(self, key):
        self.kv.pop(key, None)
        self.lists.pop(key, None)
        self.expiry.pop(key, None)

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    async def ltrim(self, key, start, end):
        items = self.lists.get(key, [])
        if end == -1:
            self.lists[key] = items[start:]
        else:
            self.lists[key] = items[start : end + 1]

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        if end == -1:
            return items[start:]
        return items[start : end + 1]

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def brpop(self, key, timeout=0):
        items = self.lists.get(key, [])
        if items:
            return key, items.pop()
        return None

    async def scan_iter(self, match=None):
        keys = set(self.kv) | set(self.lists)
        prefix = match.rstrip("*") if match and match.endswith("*") else None
        for key in sorted(keys):
            if prefix is None or key.startswith(prefix):
                yield key

    async def close(self):
        return None


def _setup():
    fake = FakeRedis()
    main.redis_client = fake
    main.app.state.redis = fake
    cfg = main.get_config()
    cfg.update(
        {
            "audit_db_enabled": False,
            "database_url": None,
            "app_base_url": "http://localhost:8024",
            "report_job_max_attempts": 2,
            "report_job_backoff_seconds": 1,
            "report_delivery_provider": "smtp",
            "smtp_from": "noreply@example.com",
            "smtp_host": "smtp.example.com",
            "admin_api_key": "adminkey",
            "csrf_token_ttl_seconds": 3600,
            "rate_limit_window_seconds": 60,
            "rate_limit_report_requests": 240,
            "rate_limit_management_requests": 240,
        }
    )
    main.app.state.config = cfg
    main.app.state.shutting_down = False
    main.app.state.runtime["scheduler_last_heartbeat"] = main.now_iso()
    main.app.state.runtime["worker_last_heartbeat"] = main.now_iso()
    return fake


def _create_site(client):
    response = client.post(
        "/sites/acme-flow",
        json={
            "site_name": "Acme Flow",
            "primary_goal": "lead",
            "report_email": "client@example.com",
            "variants": [
                {"label": "A", "url": "https://a.example"},
                {"label": "B", "url": "https://b.example"},
            ],
        },
    )
    assert response.status_code == 200
    return response.json()["management_token"]


def test_report_window_behavior():
    config = {
        "report_timezone": "America/Los_Angeles",
        "report_send_weekday": 4,
        "report_send_hour": 9,
    }
    open_dt = datetime(2026, 3, 20, 9, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    closed_dt = datetime(2026, 3, 20, 8, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    is_open, week_id = main.report_window_is_open(config, now_local=open_dt)
    assert is_open is True
    assert week_id == "2026-W12"
    is_open_closed, _ = main.report_window_is_open(config, now_local=closed_dt)
    assert is_open_closed is False


def test_resend_last_is_payload_invariant():
    _setup()
    client = TestClient(main.app)
    token = _create_site(client)
    message_counter = {"count": 0}

    def fake_send(*_args, **_kwargs):
        message_counter["count"] += 1
        return {
            "provider": "smtp",
            "provider_message_id": f"smtp-msg-{message_counter['count']}",
            "provider_status": "accepted",
        }

    main.send_report_email = fake_send

    enqueue = asyncio.run(
        main.enqueue_report_job(
            site_id="acme-flow",
            mode="test",
            week_id="2026-W12",
            source="test",
            report_email="client@example.com",
        )
    )
    assert enqueue["status"] == "queued"
    first_job = asyncio.run(main.redis_client.brpop(main.report_jobs_queue_key()))
    asyncio.run(main.process_report_job(json.loads(first_job[1])))

    dashboard = client.get(f"/dashboard/acme-flow?token={token}")
    assert dashboard.status_code == 200
    csrf = re.search(r"const csrfToken = '([^']+)'", dashboard.text).group(1)
    resend = client.post(
        f"/reports/acme-flow/weekly-summary/resend-last?token={token}",
        headers={"X-AAR-CSRF": csrf},
    )
    assert resend.status_code == 200
    second_job = asyncio.run(main.redis_client.brpop(main.report_jobs_queue_key()))
    asyncio.run(main.process_report_job(json.loads(second_job[1])))

    deliveries = client.get(f"/reports/acme-flow/deliveries?token={token}")
    assert deliveries.status_code == 200
    rows = deliveries.json()["deliveries"]
    sent_rows = [row for row in rows if row.get("status") == "sent"]
    assert len(sent_rows) >= 2
    assert sent_rows[0]["payload_hash"] == sent_rows[1]["payload_hash"]


def test_auth_and_request_id_on_ops_endpoints():
    _setup()
    client = TestClient(main.app)
    forbidden = client.get("/ops/metrics")
    assert forbidden.status_code == 403

    allowed = client.get("/ops/metrics?token=adminkey")
    assert allowed.status_code == 200
    assert allowed.headers.get("X-Request-ID")
    body = allowed.json()
    assert "send_success_rate" in body
    assert "queue_depth" in body


def test_live_ready_health_endpoints():
    _setup()
    client = TestClient(main.app)
    live = client.get("/live")
    assert live.status_code == 200
    assert live.json()["status"] == "alive"

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
