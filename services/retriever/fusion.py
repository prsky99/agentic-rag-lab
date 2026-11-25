"""RRF 융합 (3.2절).

    score(d) = Σ w_i / (k + rank_i(d))

순위만 쓰고 점수 크기는 쓰지 않습니다. 검색기마다 점수 분포가 달라서, 점수를
그대로 더하면 한쪽이 다른 쪽을 삼킵니다.
"""
from __future__ import annotations

from collections import defaultdict


def rrf(rankings: list[list[str]], k: int = 60,
        weights: list[float] | None = None) -> list[tuple[str, float]]:
    """rankings: 각 검색기의 문서 ID 순위 리스트(상위부터).
    score(d) = Σ w_i / (k + rank_i(d)), rank는 1부터."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
