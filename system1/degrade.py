"""결정 모델 장애 시 대체 판정 (4.6절).

G2 충분성은 fail-closed가 기본입니다. ``GatePolicy.on_unavailable``이
``degrade``이면 BM25 점수 하한으로 대체 판정하고, 하한을 못 넘으면 기권합니다.
대체 판정 결과에는 항상 ``fallback: True``를 남깁니다.
"""
from __future__ import annotations

from system1.client import DecisionUnavailable
from system1.gates import answerability
from system1.policy import GatePolicy

BM25_FLOOR = 12.0   # 골든셋에서 '충분' 라벨의 최저 점수 분포로 정합니다(예시값)


def answerability_with_fallback(client, question, passages, bm25_scores,
                                policy: GatePolicy) -> dict:
    try:
        return answerability(client, question, passages, policy.answer_floor)
    except DecisionUnavailable:
        if policy.on_unavailable == "abstain":
            return {"verdict": "abstain", "p_yes": 0.0, "fallback": True}
        # 폴백: 상위 문단의 BM25 점수가 절대 하한을 넘으면 통과로 봅니다.
        top = max(bm25_scores[:3], default=0.0)
        verdict = "sufficient" if top >= BM25_FLOOR else "abstain"
        return {"verdict": verdict, "p_yes": None, "fallback": True, "bm25_top": top}
