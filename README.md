# Adaptive Ads Router

Increase Google Ads lead volume and lower CAC by automatically routing paid traffic to higher-converting landing page variants.

This MVP gives you an onboarding form, site configuration API, browser-friendly redirect routing, conversion tracking, and a lightweight dashboard in one service.

Admin surfaces are token-protected: each site gets a private owner token, and you can optionally set `ADMIN_API_KEY` for cross-site admin access.

## Who This Is For

- Performance marketers and agencies spending $5k+/month on Google Ads
- Teams running multiple landing pages but choosing winners too slowly
- Founders who want a measurable path to higher ROAS, not design guesswork

## Paid Setup

- 90-minute setup + first experiment shipped: **$499**
- Weekly optimization/reporting: **$999/mo**
- [Book setup call](https://cal.com/cwalinapj/adaptive-ads-router)
- Includes a PR to your repo plus deployment instructions.
- Includes first experiment setup (2 variants) and conversion event wiring.
- Includes a weekly report template you can reuse with your team/clients.

## 5-Minute Quickstart

```bash
git clone https://github.com/cwalinapj/adaptive-ads-router.git
cd adaptive-ads-router
cp .env.example .env
docker compose up -d --build
curl -f http://localhost:8024/health
open http://localhost:8024/
```

## MVP Flow

1. Open `http://localhost:8024/` and create a site with 2 landing page variants.
2. Save the private dashboard URL returned after setup. It includes the site owner token.
3. Send paid traffic to `GET /r/{site_id}` to route each click to the current best variant.
4. Record conversions with `GET /convert/{site_id}/{session_id}` or `POST /outcome`.
5. Review variant performance in `GET /dashboard/{site_id}?token=...`.
6. Inspect raw events at `GET /events/{site_id}?token=...` or export CSV from `GET /events/{site_id}.csv?token=...`.
7. Pull a last-7-days rollup from `GET /reports/{site_id}/daily?token=...`.
8. Open the weekly client report at `GET /reports/{site_id}/weekly-summary/html?token=...`.
9. Configure `report_email` for the site to enable scheduled weekly delivery and monitor send logs in the dashboard.

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
  - acme-demo_page_a: sessions=28, conversions=4, cvr=14.29%
  - acme-demo_page_b: sessions=32, conversions=9, cvr=28.12%
Bandit winner: not yet declared
Current regime: first_100
Stats endpoint: http://localhost:8024/stats/acme-demo
```

Traffic hits the router, the router assigns a variant, conversion events are recorded, and the bandit updates weights.

## What This MVP Includes

- A homepage onboarding form at `GET /` for creating a site without writing JSON by hand.
- Site configuration endpoints at `POST /sites/{site_id}`, `GET /sites`, and `GET /sites/{site_id}` protected by owner token or admin key.
- Browser-friendly redirect routing at `GET /r/{site_id}` with tracking params appended to the chosen destination URL.
- Conversion capture at `GET /convert/{site_id}/{session_id}` for thank-you-page or redirect flows.
- A lightweight operator dashboard at `GET /dashboard/{site_id}` protected by the site owner token.
- Raw route/conversion event export at `GET /events/{site_id}` and `GET /events/{site_id}.csv` protected by the same owner token.
- Date-filtered event inspection plus a simple daily rollup at `GET /reports/{site_id}/daily`.
- A downloadable weekly client report at `GET /reports/{site_id}/weekly-summary/html`.
- Scheduled weekly HTML email delivery to one `report_email` per site, with Redis-backed queue/worker retries, dead-letter handling, and delivery logs in the dashboard/API.
- Durable audit storage in Postgres for immutable payload snapshots + delivery status timeline (`queued`/`sent`/`provider_update`/`failed`).

## What You Need To Integrate With Real Google Ads

- Point ad traffic to `GET /r/{site_id}` (or a reverse proxy in front of it) so the router can choose the destination URL per click.
- Define conversion tracking and call `GET /convert/{site_id}/{session_id}` or `POST /outcome` from your thank-you page, backend event, or pixel/webhook bridge.
- Configure where variants live by creating a site with labeled destination URLs in the home form or `POST /sites/{site_id}`.
- Set `report_email` on the site config if you want automatic weekly client-report delivery.
- Keep the private dashboard URL or owner token returned at site creation. You need it for dashboard, stats, and site management requests.
- Today’s integration method: keep your existing pages, route the click through Adaptive Ads Router, then let it redirect the visitor to the chosen variant.
- `GET /r/{site_id}` redirects to the configured variant URL and appends `aar_site_id`, `aar_page_id`, and `aar_session_id`.
- `POST /route/{site_id}` still returns the selected `page_id`, destination `container_url`, and `session_id` if you prefer a server-to-server flow.
- `POST /outcome` records `site_id`, `page_id`, `session_id`, and `converted` (`true`/`false`).
- `GET /events/{site_id}` accepts `start=YYYY-MM-DD`, `end=YYYY-MM-DD`, and `type=route|outcome` so operators can filter raw activity before exporting.
- `GET /events/{site_id}.csv` exports the same filtered feed for spreadsheets.
- `GET /reports/{site_id}/daily` returns daily route/conversion/revenue totals and defaults to the last 7 days when no explicit range is provided.
- `GET /reports/{site_id}/weekly-summary` returns a compact weekly JSON summary with top variant, daily trend, recent wins, and recent losses.
- `GET /reports/{site_id}/weekly-summary/html` renders the same summary as a printable client-facing report.
- `GET /reports/{site_id}/deliveries` returns recent send attempts (`sent` / `failed`) so you can audit automation health.
- `GET /reports/{site_id}/dead-letter` returns exhausted report jobs for operator review.
- `POST /reports/{site_id}/weekly-summary/send-test` enqueues a report send immediately (optional `email` query override).
- `POST /reports/{site_id}/weekly-summary/resend-last` enqueues a resend of the exact last generated email payload (no recompute).
- `POST /reports/{site_id}/dead-letter/replay` replays a dead-letter job with explicit confirmation (`confirmation="REPLAY <job_id>"`).
- `POST /webhooks/report-delivery/{provider}` tracks provider lifecycle updates (`accepted`/`delivered`/`bounced`/`complained`) and surfaces them in Delivery Status.
- Dashboard-triggered POST actions require `X-AAR-CSRF` issued by `GET /dashboard/{site_id}`.
- Management/report endpoints are rate-limited (`RATE_LIMIT_*` env vars).

## Troubleshooting

- Docker daemon not running:
  - Run `docker info`; if it fails, start Docker Desktop (`open -a Docker`) and retry `./scripts/demo.sh`.
- Router healthcheck not ready:
  - Wait 10-30s, then check `curl http://localhost:8024/health`.
  - If still failing, view logs with `docker compose logs router mcp --tail=200`.
- Site created but no dashboard data:
  - Open the private dashboard URL returned at setup and confirm the site exists with `curl "http://localhost:8024/sites/<site_id>?token=<owner_token>"`.
  - Run one routed visit through `/r/<site_id>` and one conversion through `/convert/<site_id>/<session_id>` before expecting stats to move.
- Need raw click/conversion history:
  - Open `GET /events/<site_id>?token=<owner_token>` for JSON or `GET /events/<site_id>.csv?token=<owner_token>` for a spreadsheet-friendly export.
- Need a quick last-week summary:
  - Open `GET /reports/<site_id>/daily?token=<owner_token>` or use the dashboard’s `Last 7 Days` card.
- Need something client-ready:
  - Open `GET /reports/<site_id>/weekly-summary/html?token=<owner_token>` from the dashboard and share or print that page.
- Need to validate email setup right now:
  - Use the dashboard button `Send test report now` or call `POST /reports/<site_id>/weekly-summary/send-test?token=<owner_token>`.
- Need to resend the exact last report payload:
  - Use the dashboard button `Resend last report` or call `POST /reports/<site_id>/weekly-summary/resend-last?token=<owner_token>`.
- Scheduled emails not arriving:
  - Ensure the site has `report_email` set.
  - Configure `APP_BASE_URL`, `REPORT_DELIVERY_PROVIDER`, and provider credentials in `.env`.
  - For webhook updates, set `REPORT_WEBHOOK_SECRET` and send it as `X-AAR-Webhook-Secret`.
  - Confirm scheduler window with `REPORT_SEND_WEEKDAY` (`0=Monday`, `4=Friday`) and `REPORT_SEND_HOUR` in `REPORT_TIMEZONE`.
  - Check `GET /reports/<site_id>/deliveries?token=<owner_token>` for the latest failure reason.
  - If retries are exhausted, check Redis dead-letter list `report_jobs:dead`.
  - Replay safely from dashboard Dead-Letter panel or `POST /reports/<site_id>/dead-letter/replay?token=<owner_token>`.
- 403 on dashboard or site endpoints:
  - Add the site owner token as `?token=<owner_token>` or send it as `X-AAR-Token: <owner_token>`.
  - For cross-site admin access, set `ADMIN_API_KEY` in `.env` and send that value instead.
- 403 on report POST actions from custom scripts:
  - Fetch `GET /dashboard/<site_id>?token=<owner_token>` and send the emitted `csrfToken` as `X-AAR-CSRF`.
- Frequent 429 responses:
  - Tune `RATE_LIMIT_WINDOW_SECONDS`, `RATE_LIMIT_MANAGEMENT_REQUESTS`, and `RATE_LIMIT_REPORT_REQUESTS`.
- Debian package fetch flake during build:
  - Dockerfiles already use apt retries/timeouts; rerun `./scripts/demo.sh` and builds should recover from transient download failures.
- Compose warning about `version`:
  - If you see an obsolete `version` warning, it is non-blocking for Docker Compose v2.

## Book A Call / DM

- [Book a call](https://cal.com/cwalinapj/adaptive-ads-router)
- [DM on X](https://x.com/cwalinapj)
- [Landing page template](promo/Sitebuilder1.0/)

## API Snapshot

Router service runs on `:8024`:

```bash
GET  /
GET  /sites
POST /sites/{site_id}
GET  /sites/{site_id}
GET  /dashboard/{site_id}
GET  /events/{site_id}
GET  /events/{site_id}.csv
GET  /reports/{site_id}/daily
GET  /reports/{site_id}/weekly-summary
GET  /reports/{site_id}/weekly-summary/html
GET  /reports/{site_id}/deliveries
GET  /reports/{site_id}/dead-letter
POST /reports/{site_id}/weekly-summary/send-test
POST /reports/{site_id}/weekly-summary/resend-last
POST /reports/{site_id}/dead-letter/replay
POST /webhooks/report-delivery/{provider}
GET  /r/{site_id}
GET  /convert/{site_id}/{session_id}
POST /route/{site_id}
POST /outcome
POST /session
GET  /stats/{site_id}
POST /validate-diff
GET  /tombstones/{site_id}
GET  /neural-data/{site_id}
```

Optional: Docker MCP service (dev tooling) runs on `:8030`; it is not required for production routing.

Report delivery uses a durable Redis queue:
- Queue: `report_jobs:queue`
- Dead-letter: `report_jobs:dead`
- Idempotency key: `report_job:idem:{site_id}:{week_id}:{mode}`
- Provider statuses are appended through webhook callbacks and shown in dashboard Delivery Status.

Durable audit data is persisted in Postgres:
- `report_payload_snapshots`: immutable email payload snapshots + payload hash
- `report_delivery_events`: append-only delivery timeline with actor, provider message id, and status progression
- `report_provider_messages`: provider message-id binding for webhook reconciliation

## Core Model

- Thompson Sampling routes visitors by posterior conversion probability
- Regimes adjust behavior as volume grows:
  - `first_100`: high exploration
  - `middle`: reduced exploration
  - `neural`: conservative, stability-focused
- Diff enforcement restricts unsafe page changes by regime

## License

MIT
