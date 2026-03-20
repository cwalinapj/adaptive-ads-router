import os
import json
import asyncio
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "/Users/root1/adaptive-ads-router/router")
import main  # noqa: E402
from audit_store import AuditStore  # noqa: E402


class FakeRedis:
    def __init__(self):
        self.kv = {}
        self.lists = {}

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        return True

    async def setex(self, key, ttl, value):
        self.kv[key] = value

    async def delete(self, key):
        self.kv.pop(key, None)
        self.lists.pop(key, None)

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


@pytest.mark.integration
def test_postgres_persists_payloads_and_delivery_timeline():
    db_url = os.getenv("TEST_DATABASE_URL")
    if not db_url:
        pytest.skip("TEST_DATABASE_URL not configured")

    audit = AuditStore(db_url)
    audit.init()

    fake = FakeRedis()
    main.redis_client = fake
    main.app.state.redis = fake
    main.audit_store = audit
    cfg = main.get_config()
    cfg.update(
        {
            "audit_db_enabled": True,
            "database_url": db_url,
            "app_base_url": "http://localhost:8024",
            "report_job_max_attempts": 2,
            "report_job_backoff_seconds": 1,
        }
    )
    main.app.state.config = cfg

    client = TestClient(main.app)
    create = client.post(
        "/sites/acme-audit",
        json={
            "site_name": "Acme Audit",
            "primary_goal": "lead",
            "report_email": "client@example.com",
            "variants": [
                {"label": "A", "url": "https://a.example"},
                {"label": "B", "url": "https://b.example"},
            ],
        },
    )
    assert create.status_code == 200
    token = create.json()["management_token"]

    route = client.post("/route/acme-audit", json={"visitor_id": "v1", "device_type": "desktop"})
    assert route.status_code == 200
    route_data = route.json()
    outcome = client.post(
        "/outcome",
        json={
            "site_id": "acme-audit",
            "page_id": route_data["page_id"],
            "session_id": route_data["session_id"],
            "converted": True,
            "revenue": 22,
        },
    )
    assert outcome.status_code == 200

    main.send_report_email = lambda *_args, **_kwargs: {
        "provider": "smtp",
        "provider_message_id": "smtp-msg-audit",
        "provider_status": "accepted",
    }
    queued = asyncio.run(
        main.enqueue_report_job(
            site_id="acme-audit",
            mode="test",
            week_id="2026-W12",
            source="integration-test",
            report_email="client@example.com",
        )
    )
    assert queued["status"] == "queued"
    popped = asyncio.run(fake.brpop(main.report_jobs_queue_key()))
    asyncio.run(main.process_report_job(json.loads(popped[1])))

    # Force read-path from Postgres by clearing Redis log/payload cache.
    asyncio.run(fake.delete(main.delivery_log_key("acme-audit")))
    asyncio.run(fake.delete(main.last_report_payload_key("acme-audit")))

    deliveries = client.get(f"/reports/acme-audit/deliveries?token={token}")
    assert deliveries.status_code == 200
    events = deliveries.json()["deliveries"]
    assert len(events) >= 1
    assert events[0]["provider_message_id"] == "smtp-msg-audit"
    assert events[0]["provider_status"] == "accepted"
    assert events[0].get("payload_hash")

    payload = asyncio.run(main.get_last_report_payload("acme-audit"))
    assert payload is not None
    assert payload["site_id"] == "acme-audit"
    assert payload.get("payload_hash")
