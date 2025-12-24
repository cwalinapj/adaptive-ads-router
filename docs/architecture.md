# Architecture

## Core Philosophy

The system is lineage-aware.

Pages do not reset; they evolve.
Failures are preserved as tombstones.
Sessions are preserved as neural DNA.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAFFIC FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROUTER (Port 8024)                                 │
│  Thompson Sampling Bandit | Regime Detection | Diff Enforcement             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌──────────┐    ┌──────────┐    ┌──────────┐
             │ Page A   │    │ Page B   │    │ Page C   │
             └──────────┘    └──────────┘    └──────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DOCKER MCP (Port 8030)                                │
│  Container Lifecycle | Health Checks | Tombstone Manager                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REDIS                                           │
│  Bandit State | Session Data | Neural States | Tombstones                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### Router (Port 8024)
- Routes traffic via Thompson Sampling
- Detects regime (first_100, middle, neural)
- Records outcomes and updates bandit
- Validates diffs before mutations
- Ingests neural states for LAM

### Docker MCP (Port 8030)
- Creates/stops page containers
- Health checks
- Tombstone management

### Redis
- `bandit:{site_id}` - Bandit state
- `session:{session_id}` - Session data
- `neural:{site_id}` - Neural states list
- `tombstones:{site_id}` - Tombstone records

## Regimes

| Regime | Sessions | Max Diff | Behavior |
|--------|----------|----------|----------|
| first_100 | 0-100 | 0.8 | Aggressive exploration |
| middle | 100-1000 | 0.4 | Balanced |
| neural | >1000 | 0.1 | Neural-guided |

## Lineage

Every page has ancestry. When a page dies:
1. Tombstone captures the failure mode
2. Successor inherits winning traits
3. Ghost memory prevents repeat mistakes

Every session leaves DNA:
1. Behavioral signals (dwell, scroll, clicks)
2. Intent scores
3. Conversion outcomes

The LAM learns from this accumulated wisdom.
