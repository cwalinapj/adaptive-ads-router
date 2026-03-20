import json
import asyncio
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, "/Users/root1/adaptive-ads-router/router")
import main  # noqa: E402


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


def _setup_app():
    fake = FakeRedis()
    main.redis_client = fake
    main.app.state.redis = fake
    cfg = main.get_config()
    cfg.update(
        {
            "app_base_url": "http://localhost:8024",
            "report_job_max_attempts": 2,
            "report_job_backoff_seconds": 1,
            "report_delivery_provider": "sendgrid",
            "report_webhook_secret": "whsec_test",
        }
    )
    main.app.state.config = cfg
    return fake


def test_send_logs_provider_accept_and_webhook_update():
    _setup_app()
    client = TestClient(main.app)
    create = client.post(
        "/sites/acme-demo",
        json={
            "site_name": "Acme Demo",
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

    # Make sure report generation has data.
    route = client.post("/route/acme-demo", json={"visitor_id": "v1", "device_type": "desktop"})
    assert route.status_code == 200
    route_data = route.json()
    outcome = client.post(
        "/outcome",
        json={
            "site_id": "acme-demo",
            "page_id": route_data["page_id"],
            "session_id": route_data["session_id"],
            "converted": True,
            "revenue": 11,
        },
    )
    assert outcome.status_code == 200

    # Fake provider send result.
    main.send_report_email = lambda *_args, **_kwargs: {
        "provider": "sendgrid",
        "provider_message_id": "sg-msg-123",
        "provider_status": "accepted",
    }
    q = asyncio.run(
        main.enqueue_report_job(
            site_id="acme-demo",
            mode="test",
            week_id="2026-W12",
            source="test",
            report_email="client@example.com",
        )
    )
    assert q["status"] == "queued"
    queued = asyncio.run(main.redis_client.brpop(main.report_jobs_queue_key()))
    asyncio.run(main.process_report_job(json.loads(queued[1])))

    deliveries = client.get(f"/reports/acme-demo/deliveries?token={token}")
    assert deliveries.status_code == 200
    first = deliveries.json()["deliveries"][0]
    assert first["provider"] == "sendgrid"
    assert first["provider_status"] == "accepted"
    assert first["provider_message_id"] == "sg-msg-123"

    # Webhook status progression.
    hook = client.post(
        "/webhooks/report-delivery/sendgrid",
        headers={"X-AAR-Webhook-Secret": "whsec_test"},
        json=[{"event": "delivered", "sg_message_id": "sg-msg-123"}],
    )
    assert hook.status_code == 200
    body = hook.json()
    assert body["updated"] == 1

    deliveries2 = client.get(f"/reports/acme-demo/deliveries?token={token}")
    latest = deliveries2.json()["deliveries"][0]
    assert latest["status"] == "provider_update"
    assert latest["provider_status"] == "delivered"
    assert latest["provider_message_id"] == "sg-msg-123"


def test_webhook_rejects_without_secret():
    _setup_app()
    client = TestClient(main.app)
    response = client.post("/webhooks/report-delivery/sendgrid", json=[{"event": "delivered"}])
    assert response.status_code == 401
