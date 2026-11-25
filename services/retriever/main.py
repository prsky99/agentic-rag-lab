"""검색 서비스 HTTP 표면 (3.5절)."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.retriever.scope import TenantScope  # noqa: E402
from services.retriever.service import BACKEND, search, warm  # noqa: E402

app = FastAPI(title="rag-retriever")


class SearchRequest(BaseModel):
    query: str
    tenant: str
    top_k: int = 50


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/ready")
def ready() -> dict:
    # Qdrant 백엔드는 임베딩 모델을 읽은 뒤에 준비 상태가 됩니다(모델 없음·미지원이면 503).
    try:
        warm()
    except Exception as exc:                     # noqa: BLE001
        raise HTTPException(503, f"embedding model not ready: {exc}") from exc
    return {"ok": True}


@app.post("/search")
def do_search(req: SearchRequest) -> dict:
    try:
        scope = TenantScope(req.tenant)          # 빈 테넌트는 여기서 거부됩니다
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    results = search(scope, req.query, top_k=req.top_k)
    return {"results": results, "backend": BACKEND}   # RETRIEVER_BACKEND 기준(local | qdrant)
