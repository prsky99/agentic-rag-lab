"""게이트웨이 (2.5절): 인증 → 요청 ID → orchestrator 프록시."""
from __future__ import annotations

import hashlib
import os
import uuid

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from services.common.otel import current_trace_id, stage
from services.gateway.tenant import Tenant, tenant_from_auth

ORCH_URL = os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8080")
app = FastAPI(title="rag-gateway")
TENANT_SALT = os.environ.get("TENANT_HASH_SALT", "")


def tenant_hash(tenant_id: str) -> str:
    """트레이스·로그에는 테넌트 원문 대신 해시만 남깁니다 (11.7절, obs/semconv.py RAG_TENANT)."""
    return hashlib.sha256((TENANT_SALT + tenant_id).encode("utf-8")).hexdigest()[:16]


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/ready")
def ready() -> dict:
    return {"ok": True}


@app.post("/v1/answer")
async def answer(request: Request, tenant: Tenant = Depends(tenant_from_auth)):
    try:
        body = await request.json()                 # 깨진 JSON·비 UTF-8 본문은 500이 아니라 400
    except ValueError:                              # JSONDecodeError·UnicodeDecodeError 모두
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    question = str(body.get("question", ""))[:4000]
    if not question:
        return JSONResponse({"error": "question required"}, status_code=400)
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
    with stage("rag.request", **{"rag.tenant": tenant_hash(tenant.id)}):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{ORCH_URL}/v1/answer",
                json={"question": question, "tenant_id": tenant.id, "tier": tenant.tier},
                headers={"x-request-id": request_id,
                         "x-internal-token": os.environ.get("INTERNAL_TOKEN", "dev")})
    if r.status_code >= 400:
        return JSONResponse({"error": "upstream", "status": r.status_code},
                            status_code=502)
    data = r.json()
    data["request_id"] = request_id
    data["trace_id"] = data.get("trace_id") or current_trace_id()
    return JSONResponse(data, headers={"x-trace-id": data["trace_id"] or ""})
