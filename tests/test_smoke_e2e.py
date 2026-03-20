import asyncio
import json
import re
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, "/Users/root1/adaptive-ads-router/router")
import main  # noqa: E402


class FakeRedis:
    def __init__(self):
        self.kv = {}
        self.lists = {}

    async def ping(self):
        return True

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        return True

    async def setex(self, key, ttl, value):
        self.kv[key] = value

    async def incr(self, key):
        current = int(self.kv.get(key, "0")) + 1
        self.kv[key] = str(current)
        return current

    async def expire(self, key, ttl):
        return True

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
            "report_delivery_provider": "sendgrid",
            "sendgrid_api_key": "sg_test",
            "smtp_from": "noreply@example.com",
            "report_webhook_secret": "whsec_smoke",
            "csrf_token_ttl_seconds": 3600,
            "rate_limit_window_seconds": 60,
            "rate_limit_report_requests": 240,
            "rate_limit_management_requests": 240,
        }
    )
    main.app.state.config = cfg
    main.app.state.shutting_down = False
    return fake


def test_end_to_end_smoke_with_fake_provider():
    _setup()
    main.send_report_email = lambda *_args, **_kwargs: {
        "provider": "sendgrid",
        "provider_message_id": "sg-msg-smoke",
        "provider_status": "accepted",
    }
    client = TestClient(main.app)

    create = client.post(
        "/sites/acme-smoke",
        json={
            "site_name": "Acme Smoke",
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

    route = client.post("/route/acme-smoke", json={"visitor_id": "v1", "device_type": "desktop"})
    assert route.status_code == 200
    route_data = route.json()
    outcome = client.post(
        "/outcome",
        json={
            "site_id": "acme-smoke",
            "page_id": route_data["page_id"],
            "session_id": route_data["session_id"],
            "converted": True,
            "revenue": 5.0,
        },
    )
    assert outcome.status_code == 200

    dashboard = client.get(f"/dashboard/acme-smoke?token={token}")
    csrf = re.search(r"const csrfToken = '([^']+)'", dashboard.text).group(1)
    send_test = client.post(
        f"/reports/acme-smoke/weekly-summary/send-test?token={token}",
        headers={"X-AAR-CSRF": csrf},
    )
    assert send_test.status_code == 200

    queued = asyncio.run(main.redis_client.brpop(main.report_jobs_queue_key()))
    asyncio.run(main.process_report_job(json.loads(queued[1])))
    deliveries = client.get(f"/reports/acme-smoke/deliveries?token={token}")
    assert deliveries.status_code == 200
    assert deliveries.json()["count"] >= 1

    webhook = client.post(
        "/webhooks/report-delivery/sendgrid",
        headers={"X-AAR-Webhook-Secret": "whsec_smoke"},
        json=[{"event": "delivered", "sg_message_id": "sg-msg-smoke"}],
    )
    assert webhook.status_code == 200
    assert webhook.json()["updated"] == 1
