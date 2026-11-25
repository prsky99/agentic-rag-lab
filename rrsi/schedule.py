"""어닐링 편집 예산 b_t (7.3절).

라운드가 진행될수록 한 후보에 넣는 편집 수를 줄입니다. 초반에는 크게 탐색하고,
후반에는 현재 최고점 근처에서 미세 조정합니다.
"""
from __future__ import annotations

import math


def edit_budget(t: int, T: int, b_min: int = 1, b_max: int = 3) -> int:
    """라운드 t(0..T-1)의 편집 예산. 첫 라운드 b_max, 마지막 라운드(t=T-1) b_min.

    분모를 T가 아니라 T-1로 둡니다. 라운드는 0..T-1까지만 실행되고 올림(ceil)을 쓰므로,
    분모가 T이면 마지막 라운드 예산도 b_min보다 크게 남습니다.
    """
    if T <= 1:
        return b_min
    ratio = 0.5 * (1 + math.cos(math.pi * min(t, T - 1) / (T - 1)))
    return max(b_min, min(b_max, math.ceil(b_min + (b_max - b_min) * ratio)))


def annealing_step(t: int, T: int, start: float, end: float) -> float:
    """스칼라 하이퍼파라미터(예: 탐색 온도)용 같은 곡선."""
    if T <= 1:
        return end
    ratio = 0.5 * (1 + math.cos(math.pi * min(t, T) / T))
    return end + (start - end) * ratio
