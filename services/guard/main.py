"""가드 서비스 (11.4절): /check."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.guard.checks import output_guard  # noqa: E402

app = FastAPI(title="rag-guard")


class CheckRequest(BaseModel):
    answer: str
    citations: list[dict] = []
    chunks: dict[str, str] = {}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/check")
def check(req: CheckRequest) -> dict:
    action, reason = output_guard(req.answer, req.citations, req.chunks)
    return {"action": action, "reason": reason}
