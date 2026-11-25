"""평가 비용 집계 시험(2차 검토 RT06, 평가 쪽).

미측정 사용량(unknown)을 평가 비용에서 0으로 세지 않고 추정치로 채우는지 확인합니다.
진화 루프의 도메인 어댑터(``domains/rag/adapter.py``)를 쓰는 평가 쪽 시험입니다.
요청 경로 쪽 RT06 시험은 ``tests/runtime/test_deadline_usage.py``에 있습니다.
"""
from __future__ import annotations

import json


def test_rt06_evaluation_cost_fills_unknown_with_estimate(tmp_path):
    from judge import trial_tokens
    assert trial_tokens({"total_tokens": 442, "unknown_calls": 4,
                         "unknown_estimate_tokens": 1900}) == (2342, 4)
    assert trial_tokens({"total_tokens": 2360, "unknown_calls": 0}) == (2360, 0)
    assert trial_tokens({}) == (None, 0) and trial_tokens(None) == (None, 0)

    from adapter import RagDomain
    d = RagDomain()
    tid = next(i for i, it in d.items.items() if not it["answerable"])
    out = tmp_path / "jobs" / "j" / "s0"
    out.mkdir(parents=True)
    trial = {"abstained": True, "answer": "", "citations": [], "retrieved": [], "error": None,
             "latency_ms": 10, "usage": {"total_tokens": 442, "usage_known": False,
                                         "unknown_calls": 4, "unknown_estimate_tokens": 1900}}
    (out / f"{tid}.json").write_text(json.dumps(trial), encoding="utf-8")
    per, extra = d.score(tmp_path, "j", [tid], 1)
    assert per[tid].tokens == [2342]                        # 0으로 세지 않습니다
    assert extra["usage_unknown_calls"] == 4 and extra["usage_unknown_runs"] == 1
    assert extra["cost_basis"] == "measured+estimated_unknown"
    # 미측정이 없으면 extra의 모양은 그대로입니다.
    trial["usage"] = {"total_tokens": 2360, "usage_known": True, "unknown_calls": 0}
    (out / f"{tid}.json").write_text(json.dumps(trial), encoding="utf-8")
    per, extra = d.score(tmp_path, "j", [tid], 1)
    assert per[tid].tokens == [2360] and "usage_unknown_calls" not in extra
