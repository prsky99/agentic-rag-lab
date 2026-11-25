# services/orchestrator/loop.py
"""재검색 중단 조건 (6.3절, 참조 구현).

정본은 `domains/rag/harness/gates.py`의 `should_retry`이며 판정 규칙은 같습니다.
"""
from __future__ import annotations


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def should_retry(prev: dict, cur: dict, attempts: int, max_attempts: int,
                 deadline_left_ms: int, cost_ms: int = 450) -> tuple[bool, str]:
    if attempts >= max_attempts:
        return False, "max_attempts"
    if deadline_left_ms < cost_ms:
        return False, "deadline"
    delta_p = cur["p_yes"] - prev["p_yes"]
    if delta_p < 0.10:
        return False, "no_gain"
    if jaccard(prev["ids"], cur["ids"]) > 0.80:
        return False, "same_evidence"
    return True, "retry"
