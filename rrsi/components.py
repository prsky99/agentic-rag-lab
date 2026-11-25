"""diff → 컴포넌트 태그와 novelty (7.3절, 8.6절).

제안자는 "무엇을 바꿀지"를 컴포넌트 단위로 말해야 하고, 이력은 "어디를
얼마나 탐색했는지"를 컴포넌트 단위로 집계합니다. 둘의 단위가 같아야 정체를
피하는 지시(미탐색 컴포넌트로 이동)가 의미가 있습니다.
"""
from __future__ import annotations

import re


def tag_diff(diff: str, signals: list[tuple[str, list[str]]]) -> set[str]:
    tags: set[str] = set()
    for component, patterns in signals:
        if any(re.search(p, diff) for p in patterns):
            tags.add(component)
    return tags or {"control_flow"}


def changed_lines(diff: str) -> tuple[int, int]:
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed


def novelty(components: set[str], history: list[dict], window: int = 3) -> float:
    """최근 window 라운드에서 건드리지 않은 컴포넌트 비율."""
    if not components:
        return 0.0
    recent: set[str] = set()
    for rec in history[-window:]:
        recent |= set(rec.get("components") or [])
    fresh = components - recent
    return len(fresh) / len(components)
