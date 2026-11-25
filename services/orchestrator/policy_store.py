# services/orchestrator/policy_store.py
"""정책 스냅샷과 카나리 선택 (6.8절, 참조 구현).

정본 하네스에서는 같은 값이 `domains/rag/harness/config.yaml`의 `gates`·`policy`에 있습니다.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicySnapshot:
    version: str
    route_floor: float
    answer_floor: float
    faith_floor: float
    rewrite_max: int


# 책 전체 기본값(config.yaml gates와 같습니다): G1 0.60 / G2 0.65 / G3 0.70, 재작성 1회
DEFAULT_POLICY = PolicySnapshot("rag-harness-0.1.0", 0.60, 0.65, 0.70, 1)


def bucket_of(key: str) -> int:
    """0~99 버킷. 내장 hash()는 프로세스마다 값이 달라지므로 안정적인 해시를 씁니다."""
    return zlib.crc32(key.encode("utf-8")) % 100


def select_policy(tenant_id: str, bucket: int, live: PolicySnapshot,
                  canary: PolicySnapshot | None, canary_pct: int = 5) -> PolicySnapshot:
    if canary and (bucket % 100) < canary_pct:
        return canary
    return live
