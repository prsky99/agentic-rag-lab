"""RED + 도메인 지표 (10.3절).

``record()``는 파이프라인 결과(domains/rag/harness/pipeline.py의 ``_result``)를 받습니다.
usage 키(gate_tokens/input_tokens/output_tokens)를 stage 라벨로 바꿔 기록하고,
합계(total_tokens)는 중복이므로 집계에서 제외합니다.
"""
from __future__ import annotations

BUCKETS = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 15]     # p95 목표 3초에 맞춘 버킷(초)
STAGES = {"gate_tokens": "gate", "input_tokens": "input", "output_tokens": "output"}

try:
    from opentelemetry import metrics

    m = metrics.get_meter("rag")
    req = m.create_counter("rag_requests_total")
    # opentelemetry-api 1.29에는 explicit_bucket_boundaries_advisory 인자가 없어 TypeError가
    # 나고 지표 전체가 꺼졌습니다. 버킷(BUCKETS)은 services/common/otel.py의 View가 정합니다.
    lat = m.create_histogram("rag_request_duration_seconds", unit="s")
    tok = m.create_counter("rag_tokens_total")
    gate = m.create_counter("rag_gate_verdicts_total")
except Exception:                                   # noqa: BLE001
    m = req = lat = tok = gate = None


def verdict_of(d: dict) -> str:
    v = d.get("verdict") or d.get("choice") or d.get("path")
    if v is None and "ok" in d:
        v = "ok" if d["ok"] else "fail"
    return str(v or "unknown")


def record(res: dict, route: str, status: int, tenant_tier: str) -> None:
    if m is None:
        return
    req.add(1, {"route": route, "status": str(status)})
    lat.record(res["latency_ms"] / 1000, {"route": route})
    for key, n in (res.get("usage") or {}).items():
        if key in STAGES and n:          # total_tokens는 합계라 집계에서 제외합니다
            tok.add(n, {"stage": STAGES[key], "tenant_tier": tenant_tier})
    for g, v in (res.get("decisions") or {}).items():
        gate.add(1, {"gate": g, "verdict": verdict_of(v)})
