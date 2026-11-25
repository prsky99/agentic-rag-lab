"""선택 규칙 (7.3절, Algorithm 2).

순서: 비교 조건(comparability) → 하드 가드 → 누락 비율 → 노이즈 하한(S* - δ) → 비용 규칙 → argmax.
δ는 "같은 하네스를 두 번 측정해도 생기는 차이"의 상한입니다. 이보다 작은 상승은
증거가 아니라 잡음이므로 탈락시킵니다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .domain import EvalSummary


@dataclass
class Decision:
    accepted: bool
    reason: str
    delta_S: float = 0.0
    delta_C: float = 0.0


def cost_rule(delta_S: float, delta_C: float, nov: float, delta: float,
              cfg: dict) -> tuple[bool, str]:
    """비용 규칙: δ를 넘는 상승이면 비용 증가를 예산 안에서 허용합니다.

    δ를 넘지 못하면 모양 점수(shaped)로 판단합니다. 토큰을 줄이는 편집이
    성능이 조금 낮아져도 순이득이면 탐색할 가치가 있습니다.
    """
    if delta_S > delta:
        budget = cfg.get("beta0", 0.03) + cfg.get("beta1", 10.0) * delta_S
        ok = delta_C <= budget
        return ok, f"budget {budget:.3f} (dC {delta_C:+.4f})"
    cost_eps = cfg.get("cost_eps", 1e-3)            # 비용 변화 0.1% 미만은 측정 잡음으로 봅니다
    if delta_S <= 1e-9 and delta_C >= -cost_eps:
        # 아무 효과도 없는 편집. novelty가 있어도 탈락시킵니다.
        return False, "no_effect (dS<=0, dC>=0)"
    shaped = (cfg.get("w_s", 0.0) * delta_S
              - cfg.get("w_c", 12.0) * delta_C
              + cfg.get("w_n", 0.5) * nov)
    return shaped > 0, f"shaped {shaped:+.4f}"


def comparability(incumbent: EvalSummary, candidate: EvalSummary) -> list[str]:
    """두 평가가 같은 조건에서 나왔는지 확인합니다(F01). 비어 있으면 비교해도 됩니다.

    문항·반복 수·평가기·심판·골든셋·고정 실행 조건(``conditions``)이 하나라도 다르거나 없으면
    점수 차이는 하네스의 효과가 아니므로 비교하지 않습니다. 후보가 바꿀 수 있는 하네스 값은
    ``conditions``에 들어 있지 않습니다(도메인이 뺍니다).
    """
    v: list[str] = []
    ic, cc = incumbent.conditions or {}, candidate.conditions or {}
    if not ic:
        v.append("incomparable: incumbent conditions missing (baseline 재등록 필요)")
    if not cc:
        v.append("incomparable: candidate conditions missing")
    if ic and cc:
        for key in sorted(set(ic) | set(cc)):
            if ic.get(key) != cc.get(key):
                v.append(f"incomparable: {key} differs")
    if incumbent.k != candidate.k:
        v.append(f"incomparable: k {candidate.k} != {incumbent.k}")
    return v


def decide(candidate: EvalSummary, incumbent: EvalSummary, delta: float,
           cfg: dict, guards: list[str], novelty: float = 0.0,
           s_star: float | None = None) -> Decision:
    dS = candidate.S - incumbent.S
    dC = ((candidate.C - incumbent.C) / incumbent.C) if incumbent.C else 0.0

    # 비교 조건이 다르면 가드보다 먼저 멈춥니다. 다른 조건의 점수 차이는 증거가 아닙니다(F01).
    blocked = comparability(incumbent, candidate)
    if blocked:
        return Decision(False, "; ".join(blocked), dS, dC)
    if guards:
        return Decision(False, "guard: " + "; ".join(guards), dS, dC)
    if candidate.missing_frac > cfg.get("invalid_missing_frac", 0.1):
        return Decision(False, f"invalid: missing {candidate.missing_frac:.1%}", dS, dC)
    # 노이즈 보정 하한: 지금까지 관측된 최고점 S*에서 δ 넘게 떨어지면 탈락
    if s_star is not None and candidate.S < s_star - delta:
        return Decision(False, f"floor: S {candidate.S:.4f} < S* {s_star:.4f} - delta {delta:.4f}",
                        dS, dC)

    ok, why = cost_rule(dS, dC, novelty, delta, cfg)
    if not ok:
        return Decision(False, f"rejected: {why}", dS, dC)
    if dS > delta:
        return Decision(True, f"admissible: gain {dS:+.4f} > delta {delta:.4f}; {why}", dS, dC)
    return Decision(True, f"cost-only: {why}", dS, dC)


def pick_best(decisions: list[tuple[str, Decision]]) -> tuple[str, Decision] | None:
    """승격 가능한 후보 중 dS가 가장 큰 하나. 동률이면 사유 문자열 순."""
    ok = [(name, d) for name, d in decisions if d.accepted]
    if not ok:
        return None
    ok.sort(key=lambda nd: (-nd[1].delta_S, nd[0]))
    return ok[0]
