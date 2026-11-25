# decision_head/metrics.py
"""ECE와 신뢰도 표 (5.5절). 텐서 메서드만 쓰므로 모듈 수준의 torch import 없이 동작합니다."""
from __future__ import annotations   # 텐서 메서드만 쓰므로 torch import 없이 동작합니다

from typing import TYPE_CHECKING

if TYPE_CHECKING:                     # 타입 표기용(실행 시에는 읽지 않음)
    import torch


def ece(probs: "torch.Tensor", correct: "torch.Tensor", bins: int = 15) -> float:
    """probs: 예측 답의 확률 [N], correct: 0/1 [N]"""
    total = len(probs)
    err = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (probs > lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        err += (m.sum() / total) * abs(probs[m].mean() - correct[m].float().mean())
    return float(err)


def reliability(probs: "torch.Tensor", correct: "torch.Tensor", bins: int = 10):
    rows = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (probs > lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                     "confidence": round(float(probs[m].mean()), 3),
                     "accuracy": round(float(correct[m].float().mean()), 3)})
    return rows
