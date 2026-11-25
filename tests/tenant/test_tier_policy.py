"""등급 정책 (11.2절): 상한·기본값·설정 덮어쓰기."""
import pytest

pytest.importorskip("jwt")
pytest.importorskip("fastapi")

from services.gateway.tenant import Tenant  # noqa: E402
from services.gateway.tier import TIERS, apply_to_config, policy_for  # noqa: E402


def test_rewrite_cap_never_exceeds_book_default():
    assert all(p.max_rewrites <= 1 for p in TIERS.values())


def test_unknown_tier_falls_back_to_free():
    assert policy_for(Tenant(id="t", tier="gold", user="u")) == TIERS["free"]


def test_apply_to_config_overrides_without_mutating():
    cfg = {"gates": {"max_rewrites": 1, "g2_floor": 0.65}, "retrieval": {"top_k": 40}, "budget": {"max_gate_calls": 5}}
    out = apply_to_config(cfg, TIERS["free"])
    assert out["gates"]["max_rewrites"] == 0
    assert out["gates"]["g2_floor"] == 0.70
    assert out["retrieval"]["top_k"] == 20
    assert cfg["retrieval"]["top_k"] == 40
