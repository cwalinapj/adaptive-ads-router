# Architecture

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
