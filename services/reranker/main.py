"""리랭커 서비스 (3.6절): /rerank."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.retriever.rerank import CrossEncoder, lexical_rank  # noqa: E402

app = FastAPI(title="rag-reranker")
_model = None


def model():
    global _model
    if _model is None and os.environ.get("RERANK_MODEL"):
        try:
            _model = CrossEncoder(os.environ["RERANK_MODEL"])
        except Exception:                              # noqa: BLE001
            _model = False
    return _model or None


class RerankRequest(BaseModel):
    query: str
    docs: list[dict]
    top_n: int = 8


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict:
    m = model()
    ranked = (m.rank(req.query, req.docs, req.top_n) if m
              else lexical_rank(req.query, req.docs, req.top_n))
    return {"results": [{"chunk_id": r.chunk_id, "score": r.score} for r in ranked]}
