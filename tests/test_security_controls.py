import json
import re
import asyncio
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, "/Users/root1/adaptive-ads-router/router")
import main  # noqa: E402


class FakeRedis:
    def __init__(self):
        self.kv = {}
        self.lists = {}
        self.expiry = {}

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
        current = int(self.kv.get(key, "0"))
        current += 1
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


def _setup(rate_limit_report=240):
    fake = FakeRedis()
    main.redis_client = fake
    main.app.state.redis = fake
    cfg = main.get_config()
    cfg.update(
        {
            "audit_db_enabled": False,
            "database_url": None,
            "rate_limit_report_requests": rate_limit_report,
            "rate_limit_management_requests": 500,
            "rate_limit_window_seconds": 60,
            "csrf_token_ttl_seconds": 3600,
        }
    )
    main.app.state.config = cfg
    return fake


def _create_site(client):
    response = client.post(
        "/sites/acme-sec",
        json={
            "site_name": "Acme Sec",
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


def test_dashboard_actions_require_csrf_header():
    _setup()
    client = TestClient(main.app)
    token = _create_site(client)

    dashboard = client.get(f"/dashboard/acme-sec?token={token}")
    assert dashboard.status_code == 200
    match = re.search(r"const csrfToken = '([^']+)'", dashboard.text)
    assert match
    csrf = match.group(1)

    missing_csrf = client.post(f"/reports/acme-sec/weekly-summary/send-test?token={token}")
    assert missing_csrf.status_code == 403

    with_csrf = client.post(
        f"/reports/acme-sec/weekly-summary/send-test?token={token}",
        headers={"X-AAR-CSRF": csrf},
    )
    assert with_csrf.status_code == 200
    assert with_csrf.json()["result"]["status"] in {"queued", "duplicate"}


def test_report_rate_limit_blocks_burst():
    _setup(rate_limit_report=1)
    client = TestClient(main.app)
    token = _create_site(client)

    first = client.get(f"/reports/acme-sec/deliveries?token={token}")
    assert first.status_code == 200
    second = client.get(f"/reports/acme-sec/deliveries?token={token}")
    assert second.status_code == 429


def test_input_sanitization_rejects_html_tags():
    _setup()
    client = TestClient(main.app)
    response = client.post(
        "/sites/acme-bad",
        json={
            "site_name": "<script>alert(1)</script>",
            "primary_goal": "lead",
            "report_email": "client@example.com",
            "variants": [
                {"label": "A", "url": "https://a.example"},
                {"label": "B", "url": "https://b.example"},
            ],
        },
    )
    assert response.status_code == 400
