# services/orchestrator/deadline.py
"""데드라인 전파 (6.5절, 참조 구현). 정본 하네스 `pipeline.Deadline`과 같은 코드입니다.

``total_ms``가 0이면 시작부터 만료입니다(기본값으로 바꾸지 않습니다). ``expired()``는 호출
전뿐 아니라 호출 뒤와 최종 반환 직전에도 검사합니다. 진행 중인 호출을 끊는 일은 각 호출의
타임아웃이 맡고, 이 객체는 남은 시간을 알려 주고 만료 여부를 판정합니다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Deadline:
    started: float
    total_ms: int

    @classmethod
    def start(cls, total_ms: int) -> "Deadline":
        return cls(time.monotonic(), total_ms)

    def left_ms(self) -> int:
        return max(0, self.total_ms - int((time.monotonic() - self.started) * 1000))

    def slice_ms(self, budget_ms: int, reserve_ms: int = 0) -> int:
        """이 단계에 쓸 수 있는 시간. 뒤 단계를 위해 reserve를 남깁니다."""
        return max(0, min(budget_ms, self.left_ms() - reserve_ms))

    def expired(self) -> bool:
        return self.left_ms() <= 0
