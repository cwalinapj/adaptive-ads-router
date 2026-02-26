# Adaptive Ads Router

Increase Google Ads lead volume and lower CAC by automatically routing paid traffic to higher-converting landing page variants.

## Who This Is For

- Performance marketers and agencies spending $5k+/month on Google Ads
- Teams running multiple landing pages but choosing winners too slowly
- Founders who want a measurable path to higher ROAS, not design guesswork

## 5-Minute Quickstart

```bash
git clone https://github.com/cwalinapj/adaptive-ads-router.git
cd adaptive-ads-router
cp .env.example .env
docker compose up -d --build
curl http://localhost:8024/health
```

## One-Command Demo (Sample Data)

```bash
./scripts/demo.sh
```

The demo starts the stack, runs simulated paid traffic, records conversions, and prints variant performance.

Example output:

```text
=== Adaptive Ads Router Demo ===
Router URL: http://localhost:8024
Site ID: acme-demo
Simulated sessions: 60
Variant summary:
  - acme-demo_page_a: sessions=34, conversions=6, cvr=17.65%
  - acme-demo_page_b: sessions=26, conversions=5, cvr=19.23%
Bandit winner: not yet declared
Current regime: first_100
Stats endpoint: http://localhost:8024/stats/acme-demo
```

## Paid Offers (Monetization CTA)

### 1) Paid Setup + Optimization Package ($2,500 fixed)

- Instrumentation + router setup
- Initial experiment design and guardrails
- One conversion-focused iteration sprint

### 2) Managed Experiments + Weekly Reporting ($1,500/month)

- Continuous test queue and variant rotation
- Weekly performance report (CVR, CAC proxy, winner confidence)
- Ongoing guardrail checks and execution support

## Book A Call / DM

- [Book a call](https://cal.com/cwalinapj/adaptive-ads-router)
- [DM on X](https://x.com/cwalinapj)
- [Landing page template](promo/Sitebuilder1.0/index.html)

## API Snapshot

Router service runs on `:8024`:

```bash
POST /route/{site_id}
POST /outcome
POST /session
GET  /stats/{site_id}
POST /validate-diff
GET  /tombstones/{site_id}
GET  /neural-data/{site_id}
```

Docker MCP service runs on `:8030`.

## Core Model

- Thompson Sampling routes visitors by posterior conversion probability
- Regimes adjust behavior as volume grows:
  - `first_100`: high exploration
  - `middle`: reduced exploration
  - `neural`: conservative, stability-focused
- Diff enforcement restricts unsafe page changes by regime

## License

MIT
