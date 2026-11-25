"""게이트 임계값 스윕과 캘리브레이션 확인 (4.5절).

'충분함'을 양성으로 둡니다.
- FP: 불충분한데 통과시킴 → 환각 위험
- FN: 충분한데 막음 → 기권 손실
기본 가중 ``cost_fp=4, cost_fn=1``은 "환각 1건이 기권 4건만큼 나쁘다"는 사업 판단입니다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateCase:
    qid: str
    p_yes: float          # 게이트가 낸 확률
    label: bool           # 사람이 라벨링한 정답(충분한가?)


def sweep(cases: list[GateCase], cost_fp: float = 4.0, cost_fn: float = 1.0):
    """임계값 0.30~0.95를 훑어 가중 비용이 최소인 지점을 고릅니다.
    비용이 같으면 먼저 나온(작은) 임계값, 즉 recall이 높은 쪽을 고릅니다."""
    rows = []
    for t in [i / 100 for i in range(30, 96)]:
        tp = sum(1 for c in cases if c.p_yes >= t and c.label)
        fp = sum(1 for c in cases if c.p_yes >= t and not c.label)
        fn = sum(1 for c in cases if c.p_yes < t and c.label)
        tn = sum(1 for c in cases if c.p_yes < t and not c.label)
        cost = fp * cost_fp + fn * cost_fn
        rows.append({"t": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": tp / max(tp + fp, 1),
                     "recall": tp / max(tp + fn, 1),
                     "cost": cost})
    best = min(rows, key=lambda r: r["cost"])
    return rows, best


def reliability(cases: list[GateCase], bins: int = 10) -> list[dict]:
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        xs = [c for c in cases if lo < c.p_yes <= hi]
        if not xs:
            continue
        conf = sum(c.p_yes for c in xs) / len(xs)
        acc = sum(1 for c in xs if c.label) / len(xs)
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(xs),
                    "confidence": round(conf, 3), "accuracy": round(acc, 3)})
    return out


def ece(cases: list[GateCase], bins: int = 15) -> float:
    total = len(cases)
    if not total:
        return 0.0
    err = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        xs = [c for c in cases if lo < c.p_yes <= hi]
        if not xs:
            continue
        conf = sum(c.p_yes for c in xs) / len(xs)
        acc = sum(1 for c in xs if c.label) / len(xs)
        err += len(xs) / total * abs(conf - acc)
    return err
