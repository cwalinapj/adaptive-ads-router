#!/usr/bin/env python3
"""Run a sample conversion experiment against Adaptive Ads Router."""

import json
import os
import random
import sys
import urllib.error
import urllib.request


ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8024").rstrip("/")
SITE_ID = os.getenv("DEMO_SITE_ID", "acme-demo")
ROUNDS = int(os.getenv("DEMO_ROUNDS", "60"))
SEED = int(os.getenv("DEMO_SEED", "42"))


def request_json(method, path, payload=None):
    url = f"{ROUTER_URL}{path}"
    headers = {}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, method=method, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def main():
    if len(sys.argv) > 1:
        rounds = int(sys.argv[1])
    else:
        rounds = ROUNDS

    rng = random.Random(SEED)
    summary = {}

    try:
        request_json("POST", f"/reset/{SITE_ID}")
    except urllib.error.URLError as exc:
        print(f"Cannot reach router at {ROUTER_URL}: {exc}", file=sys.stderr)
        return 1

    for idx in range(rounds):
        route = request_json(
            "POST",
            f"/route/{SITE_ID}",
            {
                "visitor_id": f"visitor_{idx + 1}",
                "device_type": "mobile",
                "utm_source": "google",
                "utm_campaign": "demo_paid_search",
            },
        )
        page_id = route["page_id"]
        session_id = route["session_id"]

        if page_id.endswith("_a"):
            conversion_rate = 0.10
        else:
            conversion_rate = 0.22

        converted = rng.random() < conversion_rate
        request_json(
            "POST",
            "/outcome",
            {
                "site_id": SITE_ID,
                "page_id": page_id,
                "session_id": session_id,
                "converted": converted,
                "revenue": 97.0 if converted else 0.0,
            },
        )

        if page_id not in summary:
            summary[page_id] = {"sessions": 0, "conversions": 0}
        summary[page_id]["sessions"] += 1
        summary[page_id]["conversions"] += int(converted)

    stats = request_json("GET", f"/stats/{SITE_ID}")

    print("=== Adaptive Ads Router Demo ===")
    print(f"Router URL: {ROUTER_URL}")
    print(f"Site ID: {SITE_ID}")
    print(f"Simulated sessions: {rounds}")
    print("Variant summary:")
    for page_id in sorted(summary.keys()):
        sessions = summary[page_id]["sessions"]
        conversions = summary[page_id]["conversions"]
        print(
            f"  - {page_id}: sessions={sessions}, conversions={conversions}, "
            f"cvr={pct(conversions, sessions):.2f}%"
        )

    winner = stats.get("winner")
    if winner:
        print(f"Bandit winner: {winner[0]} (confidence {winner[1]:.2f})")
    else:
        print("Bandit winner: not yet declared")

    print(f"Current regime: {stats.get('regime')}")
    print(f"Stats endpoint: {ROUTER_URL}/stats/{SITE_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
