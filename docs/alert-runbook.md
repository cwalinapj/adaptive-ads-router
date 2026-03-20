# Adaptive Ads Router Alert Runbook

This runbook covers the three production alerts:

- `scheduler_stalled`
- `failure_spike`
- `zero_send_day`

Assumes local/docker deployment with router on `http://localhost:8024`.

## Common Triage Commands

```bash
# Liveness/readiness
curl -fsS http://localhost:8024/live
curl -fsS http://localhost:8024/ready
curl -fsS http://localhost:8024/health

# Current alerts and metrics (admin token required)
curl -fsS "http://localhost:8024/ops/alerts?token=$ADMIN_API_KEY"
curl -fsS "http://localhost:8024/ops/metrics?token=$ADMIN_API_KEY"

# Router logs
docker compose logs router --tail=200
```

## Alert: scheduler_stalled

### Meaning
Scheduler heartbeat has not advanced within `ALERT_SCHEDULER_STALL_SECONDS`.

### First Checks
```bash
curl -fsS "http://localhost:8024/ops/alerts?token=$ADMIN_API_KEY"
docker compose ps router
```

### Likely Causes
- Router process restarted repeatedly.
- Event loop blocked by long-running operation.
- Redis unavailable causing scheduler loop failures.

### Remediation
```bash
# Inspect redis connectivity and router health
docker compose logs redis --tail=100
docker compose logs router --tail=200

# Restart router if heartbeat is frozen
docker compose restart router

# Confirm clear
curl -fsS "http://localhost:8024/ops/alerts?token=$ADMIN_API_KEY"
```

## Alert: failure_spike

### Meaning
`failed / (sent + failed)` exceeded `ALERT_FAILURE_SPIKE_RATIO` with at least `ALERT_FAILURE_SPIKE_MIN_EVENTS` recent events.

### First Checks
```bash
curl -fsS "http://localhost:8024/ops/metrics?token=$ADMIN_API_KEY"
curl -fsS "http://localhost:8024/ops/alerts?token=$ADMIN_API_KEY"
```

### Likely Causes
- Delivery provider auth/config invalid.
- Provider outage or throttling.
- Invalid recipient addresses.

### Remediation
```bash
# Inspect send and provider update logs
docker compose logs router --tail=300 | rg -n "report|provider|failed|webhook"

# Verify provider config in environment
rg -n "REPORT_DELIVERY_PROVIDER|SMTP_|SENDGRID_API_KEY|POSTMARK_SERVER_TOKEN|REPORT_WEBHOOK_SECRET" .env

# Re-run one test send from API (replace site/token)
curl -X POST "http://localhost:8024/reports/<site_id>/weekly-summary/send-test?token=<owner_token>" \
  -H "X-AAR-CSRF: <csrf_token_from_dashboard>"
```

If dead letters accumulate:

```bash
curl -fsS "http://localhost:8024/reports/<site_id>/dead-letter?token=<owner_token>"
```

Replay only with explicit confirmation:

```bash
curl -X POST "http://localhost:8024/reports/<site_id>/dead-letter/replay?token=<owner_token>" \
  -H "content-type: application/json" \
  -H "X-AAR-CSRF: <csrf_token_from_dashboard>" \
  -d '{"job_id":"<job_id>","confirmation":"REPLAY <job_id>"}'
```

## Alert: zero_send_day

### Meaning
At least one report-enabled site exists, but zero successful sends were observed in the last 24 hours.

### First Checks
```bash
curl -fsS "http://localhost:8024/ops/alerts?token=$ADMIN_API_KEY"
curl -fsS "http://localhost:8024/ops/metrics?token=$ADMIN_API_KEY"
```

### Likely Causes
- Scheduler window never opened (`REPORT_SEND_WEEKDAY`, `REPORT_SEND_HOUR`, `REPORT_TIMEZONE`).
- Sites missing `report_email` values.
- Worker disabled or unable to drain queue.

### Remediation
```bash
# Validate scheduler/worker config
rg -n "REPORT_SCHEDULER_ENABLED|REPORT_WORKER_ENABLED|REPORT_SEND_WEEKDAY|REPORT_SEND_HOUR|REPORT_TIMEZONE" .env

# Verify queue depth and worker progress
curl -fsS "http://localhost:8024/ops/metrics?token=$ADMIN_API_KEY"
docker compose logs router --tail=250 | rg -n "report_job|scheduler|worker"

# Manually trigger one report to verify end-to-end
a) open dashboard and click "Send test report now"
b) confirm delivery status appears in dashboard and /reports/<site_id>/deliveries
```

## Monitoring Stack Quickstart

```bash
# Requires main app stack running first
docker compose up -d --build

# Start Prometheus + Grafana
docker compose -f docker-compose.monitoring.yml up -d

# UIs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000
```

Grafana dashboard auto-loads as `Adaptive Ads Router - Ops`.
