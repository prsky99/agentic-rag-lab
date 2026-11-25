"""게이트 판정 로그 (2.5절, 10.4절). 한 줄 JSON으로 남깁니다."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

log = logging.getLogger("gate")


@dataclass
class GateRecord:
    trace_id: str
    tenant_id: str
    gate: str
    question_id: str
    type: str
    answer: float | str
    threshold: float | None
    verdict: str
    policy: str
    model: str
    latency_ms: int
    fallback: bool = False
    attempt: int = 1


def emit(rec: GateRecord) -> None:
    log.info(json.dumps(asdict(rec), ensure_ascii=False, separators=(",", ":")))
