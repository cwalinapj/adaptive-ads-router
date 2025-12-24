# Adaptive Ads Router

A high-velocity, container-based bandit system for Google Ads landing pages.

## Core Principles

- **One container per landing page** - Isolated testing environments
- **One LLM governor per container** - First-100 regime control
- **Divergence-gated evolution** - Changes require performance proof
- **Container Diff Enforcement** - Prevents exploration drift
- **Session-level neural ingestion** - For LAM distillation

## Regimes

| Regime | Sessions | Behavior |
|--------|----------|----------|
| **First-100** | 0-100 | Fast evolution, hard resets, LLM-gated |
| **100-1000** | 100-1000 | Ghost-memory weighting |
| **>1000** | 1000+ | Neural + LAM dominant |

## Quick Start

```bash
# Clone
git clone https://github.com/cwalinapj/adaptive-ads-router.git
cd adaptive-ads-router

# Configure
cp .env.example .env

# Run
docker-compose up -d

# Test
curl http://localhost:8024/health
curl http://localhost:8024/route/site123
```

## Components

### Router (Port 8024)

FastAPI service that routes traffic to page variants using Thompson Sampling.

```bash
# Get routing decision
POST /route/{site_id}
{
  "visitor_id": "v123",
  "device_type": "mobile"
}

# Record outcome
POST /outcome
{
  "site_id": "site123",
  "page_id": "page_a",
  "converted": true,
  "revenue": 49.99
}

# Ingest session neural state
POST /session
{
  "session_id": "sess_123",
  "site_id": "site123",
  "page_id": "page_a",
  "dwell_time": 45.2,
  "max_scroll": 0.85,
  "conversion": true
}
```

### Docker MCP (Port 8030)

Container management and diff enforcement.

```bash
# Validate container diff
POST /validate
{
  "site_id": "site123",
  "changes": {"cta_color": "#ff0000"},
  "hypothesis": "Red CTA increases urgency"
}

# Create tombstone
POST /tombstone?site_id=site123&page_id=page_a&successor_id=page_b
```

## Diff Enforcement

Each regime has limits on what changes are allowed:

| Regime | Max Diff Score | Allowed Changes |
|--------|----------------|-----------------|
| First-100 | 0.8 | CTA, headlines, hero, layout, forms |
| Middle | 0.4 | CTA, headlines, testimonials |
| Neural | 0.1 | CTA text, headlines only |

**Always Forbidden:** logo, brand_colors, legal_text, pricing

## Docker Images

Docker images are automatically built and pushed to GitHub Container Registry:

```bash
ghcr.io/cwalinapj/adaptive-ads-router-router:latest
ghcr.io/cwalinapj/adaptive-ads-router-mcp:latest
```

## License

MIT
