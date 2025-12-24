"""Docker MCP - Container Management for Adaptive Ads"""

import os
import json
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
import httpx

from schemas import (
    ContainerInfo, CreateContainerRequest, CreateContainerResponse,
    DiffProposal, DiffValidationResult
)
from validator import ContainerValidator


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True
    )
    app.state.validator = ContainerValidator()
    print("MCP started")
    yield
    await app.state.redis.close()


app = FastAPI(title="Docker MCP", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def now_iso():
    return datetime.now().isoformat()


async def get_regime(site_id: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://router:8024/stats/{site_id}", timeout=5.0)
            if resp.status_code == 200:
                return resp.json().get("regime", "first_100")
    except:
        pass
    return "first_100"


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "docker-mcp"}


@app.post("/containers", response_model=CreateContainerResponse)
async def create_container(request: CreateContainerRequest):
    container_id = f"cnt_{request.page_id}"
    info = ContainerInfo(
        container_id=container_id,
        site_id=request.site_id,
        page_id=request.page_id,
        status="running",
        url=f"http://pages.local/{request.site_id}/{request.page_id}",
        created_at=now_iso()
    )
    await app.state.redis.set(
        f"container:{request.site_id}:{request.page_id}",
        info.model_dump_json()
    )
    return CreateContainerResponse(
        container_id=container_id,
        page_id=request.page_id,
        url=info.url,
        status="running"
    )


@app.get("/containers/{site_id}/{page_id}", response_model=ContainerInfo)
async def get_container(site_id: str, page_id: str):
    data = await app.state.redis.get(f"container:{site_id}:{page_id}")
    if not data:
        raise HTTPException(404, "Container not found")
    return ContainerInfo(**json.loads(data))


@app.post("/validate", response_model=DiffValidationResult)
async def validate_diff(proposal: DiffProposal):
    regime = await get_regime(proposal.site_id)
    allowed, violations, diff_score = app.state.validator.validate(
        regime, proposal.changes, proposal.hypothesis
    )
    return DiffValidationResult(
        allowed=allowed,
        diff_score=diff_score,
        max_allowed=app.state.validator.get_max_diff_score(regime),
        violations=violations,
        reason="; ".join(violations) if violations else None
    )


@app.post("/tombstone")
async def create_tombstone(site_id: str, page_id: str, successor_id: str):
    tombstone = {
        "tombstone_id": f"tomb_{page_id}",
        "site_id": site_id,
        "terminated_page_id": page_id,
        "successor_page_id": successor_id,
        "created_at": now_iso()
    }
    await app.state.redis.rpush(f"tombstones:{site_id}", json.dumps(tombstone))
    return tombstone


@app.get("/tombstones/{site_id}")
async def get_tombstones(site_id: str, limit: int = 10):
    tombstones = await app.state.redis.lrange(f"tombstones:{site_id}", -limit, -1)
    return {"site_id": site_id, "tombstones": [json.loads(t) for t in tombstones]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)
