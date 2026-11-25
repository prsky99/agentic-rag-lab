"""필터 적용기.

검색 백엔드가 qdrant든 메모리든 같은 필터 객체를 받아 같은 방식으로 거릅니다.
격리 테스트가 백엔드와 무관하게 성립하는 이유입니다.
"""
from __future__ import annotations


def _match(payload: dict, cond) -> bool:
    match = getattr(cond, "match", None)
    value = payload.get(getattr(cond, "key", ""))
    if hasattr(match, "value"):
        return value == match.value
    if hasattr(match, "any"):
        return value in set(match.any)
    return False


def matches(payload: dict, flt) -> bool:
    if flt is None:
        return True
    for cond in getattr(flt, "must", []) or []:
        if not _match(payload, cond):
            return False
    for cond in getattr(flt, "must_not", []) or []:
        if _match(payload, cond):
            return False
    return True
