"""테넌트 등급별 정책 (11.2절).

등급(tier)은 JWT claim에서 옵니다(`tenant.py`). 등급마다 재작성 상한, 검색 상위 k,
게이트 호출 예산, 기권 기준 보정값을 다르게 둡니다. 재작성 상한은 책 전체의
결정값 `max_rewrites: 1`을 넘지 않고, 무료 등급만 0으로 낮춥니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # orchestrator가 JWT 라이브러리 없이도 읽을 수 있게 합니다
    from .tenant import Tenant


@dataclass(frozen=True)
class TierPolicy:
    max_rewrites: int          # 재작성 상한(0이면 근거 부족 시 바로 기권)
    top_k: int                 # 리랭킹 전 검색 후보 수
    max_gate_calls: int        # 요청 한 건의 결정 모델 호출 예산
    abstain_floor_delta: float  # G2 임계값 보정(+면 더 쉽게 기권)


TIERS = {
    "free":       TierPolicy(0, 20, 4, +0.05),
    "pro":        TierPolicy(1, 40, 6, 0.0),
    "enterprise": TierPolicy(1, 60, 8, -0.02),
}


def policy_for(tenant: "Tenant") -> TierPolicy:
    """알 수 없는 등급은 가장 보수적인 free 정책으로 처리합니다."""
    return policy_for_tier(tenant.tier)


def policy_for_tier(tier: str) -> TierPolicy:
    """등급 문자열(게이트웨이가 JWT에서 확정한 값)로 정책을 고릅니다."""
    return TIERS.get(str(tier), TIERS["free"])


def request_policy(cfg: dict, tier: str) -> dict:
    """요청 한 건에 넘길 정책 절(gates·retrieval·budget). 공유 설정은 바꾸지 않습니다."""
    out = apply_to_config(cfg, policy_for_tier(tier))
    return {k: out[k] for k in ("gates", "retrieval", "budget")}


def apply_to_config(cfg: dict, policy: TierPolicy) -> dict:
    """하네스 설정(config.yaml 구조)에 등급 정책을 덮어쓴 사본을 돌려줍니다."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    out.setdefault("gates", {})
    out.setdefault("retrieval", {})
    out.setdefault("budget", {})
    out["gates"]["max_rewrites"] = min(policy.max_rewrites, cfg.get("gates", {}).get("max_rewrites", 1))
    out["gates"]["g2_floor"] = round(cfg.get("gates", {}).get("g2_floor", 0.65) + policy.abstain_floor_delta, 4)
    out["retrieval"]["top_k"] = policy.top_k
    out["budget"]["max_gate_calls"] = policy.max_gate_calls
    return out
