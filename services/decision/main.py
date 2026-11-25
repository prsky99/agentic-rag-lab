"""결정 모델 프록시 decision-api (4.3절, 12.2절).

게이트는 이 서비스의 `/v1/systemone`만 봅니다. 뒤에 무엇이 있는지(같은 파드의
shim+vLLM, 외부 API)는 이 파일이 숨깁니다. 업스트림이 없으면 로컬 대체 판정으로
내려갑니다(kind·CI 기본). 업스트림 앞에 세마포어를 두고 그 대기시간을 Prometheus로
내보냅니다. KEDA는 이 값으로 스케일합니다(deploy/decision/20-keda.yaml).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "domains" / "rag"))

from harness.llm import DecisionClient  # noqa: E402

try:
    from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                                   generate_latest)
except ImportError:                       # 메트릭 라이브러리가 없어도 서비스는 뜹니다
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    class _Noop:
        def __init__(self, *a, **k): ...
        def labels(self, **k): return self
        def inc(self, *a): ...
        def dec(self, *a): ...
        def observe(self, *a): ...

    Counter = Gauge = Histogram = _Noop  # type: ignore

    def generate_latest() -> bytes:       # type: ignore
        return b""

UPSTREAM = os.environ.get("DECISION_UPSTREAM", "")       # 예: http://127.0.0.1:3001/v1 (shim)
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "32"))
SHIM_TOKEN = os.environ.get("SHIM_TOKEN", "")          # shim이 토큰을 요구할 때 함께 보냅니다
AUTH = {"authorization": f"Bearer {SHIM_TOKEN}"} if SHIM_TOKEN else {}

queue_depth = Gauge("decision_queue_depth", "requests waiting for a slot")
queue_wait = Histogram("decision_queue_wait_seconds", "time waiting for a slot",
                       buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
inflight = Gauge("decision_inflight", "requests being served")
errors = Counter("decision_errors_total", "upstream errors", ["code"])

app = FastAPI(title="rag-decision")
_local = DecisionClient(mode="local")
_sem = asyncio.Semaphore(MAX_CONCURRENCY)
_http = httpx.AsyncClient(timeout=httpx.Timeout(
    float(os.environ.get("DECISION_TIMEOUT", "2.0")), connect=0.3))


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "upstream": bool(UPSTREAM), "mode": "upstream" if UPSTREAM else "local"}


@app.post("/v1/systemone")
async def systemone(request: Request):
    if not UPSTREAM:
        body = await request.json()
        answers = _local.ask(body.get("state", {}), body.get("questions", {}))
        # 로컬 대체 판정은 토큰을 세지 않습니다(클라이언트가 미측정으로 기록합니다).
        return {"model": body.get("model", "local"), "answers": answers,
                "usage_known": False}
    body = await request.body()
    queue_depth.inc()
    t0 = time.perf_counter()
    async with _sem:
        queue_depth.dec()
        queue_wait.observe(time.perf_counter() - t0)
        inflight.inc()
        try:
            r = await _http.post(UPSTREAM.rstrip("/") + "/systemone", content=body,
                                 headers={"content-type": "application/json", **AUTH})
        except httpx.HTTPError as exc:
            errors.labels(code="unavailable").inc()
            return JSONResponse({"error": "decision_unavailable", "detail": str(exc)},
                                status_code=503)
        finally:
            inflight.dec()
    if r.status_code >= 400:
        errors.labels(code=str(r.status_code)).inc()
    # 상태 코드를 그대로 돌려줍니다. 422는 클라이언트가 재시도하지 않습니다(4.4절).
    return Response(content=r.content, status_code=r.status_code,
                    media_type="application/json")


@app.get("/v1/version")
async def version():
    if not UPSTREAM:
        return {"model": "jev-1.13.0", "mode": "local"}
    r = await _http.get(UPSTREAM.rstrip("/") + "/version", headers=AUTH)
    return Response(content=r.content, status_code=r.status_code,
                    media_type="application/json")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
