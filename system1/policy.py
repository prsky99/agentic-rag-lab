"""게이트 임계값과 판정 정책 (4.4절).

임계값은 코드에 흩어 놓지 않고 ``GatePolicy`` 하나에 모읍니다. 6장의 A/B·카나리는
이 객체를 설정에서 주입하는 방식으로 구현합니다. 기본값은 책 전체 공통값
(G1 0.60 / G2 0.65, 재작성 밴드 0.30~0.65 / G3 0.70, 재작성 1회)입니다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatePolicy:
    route_floor: float = 0.60
    answer_floor: float = 0.65
    faith_floor: float = 0.70
    rewrite_max: int = 1
    on_unavailable: str = "degrade"   # degrade(대체 판정) | abstain(즉시 기권)


def decide_answerability(p_yes: float, policy: GatePolicy) -> str:
    if p_yes >= policy.answer_floor:
        return "sufficient"
    if p_yes >= 0.30:
        return "rewrite"
    return "abstain"
