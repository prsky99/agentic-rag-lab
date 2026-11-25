"""평가 경계 시험(4차 검토 R01·R02·R03).

* R01: 운영 기권(데드라인 소진·호출 예산·구성 요소 장애)이 정상 기권 만점을 받던 문제. 실제
  RAGPipeline에서 q_0025만 deadline_ms=0으로 실행하면 reason=deadline_exceeded, error=None인데도
  reward=1.0, 문항별 퇴행 0건으로 통과했습니다.
* R02: 문항별 결과(items)가 빠진 보고서가 --item-regressions·--init-baseline을 통과하던 문제.
* R03: 정답 판정을 보상 계산과 correct 기록에서 두 번 불러, 교대로 답하는 심판에서 reward와
  correct가 어긋나던 문제. 심판 장애가 오답(0)과 구별되지 않던 문제.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "eval" / "check_thresholds.py"
BASE = ROOT / "eval" / "reports" / "base_ragas.json"
UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8", "OTEL_SDK_DISABLED": "true"}
sys.path.insert(0, str(ROOT / "eval"))

import run_subset as rs  # noqa: E402
from outcomes import ABSTAIN_REASONS, FAILURES, outcome  # noqa: E402


def _check(report: dict, tmp_path: Path, *args: str, base: Path = BASE):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return subprocess.run([sys.executable, str(CHECK), str(path), "--baseline", str(base), *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=UTF8)


@pytest.fixture(scope="module")
def env():
    from harness.llm import DecisionClient
    from harness.pipeline import RAGPipeline
    return {"items": rs.load_goldenset(), "texts": rs.chunk_texts(), "judge": DecisionClient(),
            "pipe": RAGPipeline.from_config(ROOT / "domains/rag/harness/config.yaml"),
            "ids": rs.load_subset(ROOT / "eval/ci/regression_subset.json")}


class _DeadlineFor:
    """지정한 질문에서만 deadline_ms=0으로 실행하는 파이프라인 감싸개."""

    def __init__(self, inner, question: str) -> None:
        self.inner, self.question = inner, question

    def answer_sync(self, question, tenant, tier="free", deadline_ms=None, policy=None):
        if question == self.question:
            deadline_ms = 0
        return self.inner.answer_sync(question, tenant, tier, deadline_ms, policy)


# ---------------------------------------------------------------------------
# R01 기권 사유 분류
# ---------------------------------------------------------------------------
def test_r01_reason_table_covers_every_pipeline_reason():
    """파이프라인이 내는 기권 사유를 코드에서 모아 분류표에 모두 있는지 확인합니다."""
    src = (ROOT / "domains/rag/harness/pipeline.py").read_text(encoding="utf-8")
    gates = (ROOT / "domains/rag/harness/gates.py").read_text(encoding="utf-8")
    guard = (ROOT / "domains/rag/harness/guard.py").read_text(encoding="utf-8")
    found = set(re.findall(r'reason\s*=\s*"([a-z_0-9]+)"', src))
    found |= set(re.findall(r'reason = "([a-z_0-9]+)"', src))
    found |= set(re.findall(r'"([a-z_]+_exceeded)"', src))
    seg = gates[gates.index("def should_retry"):]
    seg = seg.split("\ndef ", 1)[0]
    found |= {f"stop_{w}" for w in re.findall(r'return False, "([a-z_]+)"', seg)}
    found |= {"stop_max_attempts"}                       # 첫 재작성 상한(pipeline.py)
    seg = guard[guard.index("def output_block_reason"):]
    found |= {f"output_blocked_{w}" for w in re.findall(r'return "([a-z_]+)"', seg)}
    found -= {"no_retrieval", "decision_unavailable"}   # 답변 경로·G1 대체 사유(기권 아님)
    assert found, "사유를 찾지 못했습니다"
    missing = sorted(found - set(ABSTAIN_REASONS))
    assert not missing, f"분류표에 없는 기권 사유: {missing}"


@pytest.mark.parametrize("reason, kind", [
    ("insufficient_evidence", "abstained"), ("ungrounded", "abstained"),
    ("deadline_exceeded", "budget_exceeded"), ("call_budget_exceeded", "budget_exceeded"),
    ("stop_deadline", "budget_exceeded"), ("retrieval_unavailable", "ops_failure"),
    ("g3_unavailable", "ops_failure"), ("injection_in_question", "blocked"),
    ("output_blocked_secret_pattern", "blocked"), ("brand_new_reason", "unknown"), ("", "unknown"),
])
def test_r01_outcome_classification(reason, kind):
    assert outcome({"abstained": True, "reason": reason, "error": None}) == kind
    assert outcome({"abstained": True, "reason": reason, "error": "boom"}) == "error"


def test_r01_reward_policy_for_each_kind(env):
    from judge import answer_reward
    unans, ans = env["items"]["q_0025"], env["items"]["q_0001"]
    j = env["judge"]
    r = lambda item, reason: answer_reward(item, {"abstained": True, "reason": reason,  # noqa: E731
                                                  "error": None}, env["texts"], j)
    assert r(unans, "insufficient_evidence") == 1.0          # 정상 기권은 만점 유지
    for reason in ("deadline_exceeded", "call_budget_exceeded", "retrieval_unavailable",
                   "g3_unavailable", "something_new"):
        assert r(unans, reason) == 0.0, reason               # 운영 기권·알 수 없는 사유는 0점
    assert r(unans, "injection_in_question") == 1.0          # 답변 불가 문항의 보안 차단 = 기권 인정
    assert r(ans, "output_blocked_secret_pattern") == 0.0    # 답변 가능 문항의 차단 = 오답


@pytest.fixture(scope="module")
def deadline_report(env):
    pipe = _DeadlineFor(env["pipe"], env["items"]["q_0025"]["question"])
    return rs.evaluate(env["ids"], env["items"], env["texts"], pipe, env["judge"],
                       subset="eval/ci/regression_subset.json")


def test_r01_zero_deadline_is_budget_failure_not_abstention(deadline_report):
    row = next(r for r in deadline_report["items"] if r["id"] == "q_0025")
    assert row["reason"] == "deadline_exceeded" and row["error"] is None
    assert row["outcome"] == "budget_exceeded" and row["reward"] == 0.0
    assert deadline_report["abstain_unanswerable"] == 0.0
    assert deadline_report["error_rate"] == round(1 / 12, 4)
    assert deadline_report["outcomes"]["budget_exceeded"] == 1
    base = json.loads(BASE.read_text(encoding="utf-8"))
    assert deadline_report["S"] < base["S"]


def test_r01_zero_deadline_fails_regression_check(deadline_report, tmp_path):
    proc = _check(deadline_report, tmp_path, "--item-regressions")
    assert proc.returncode == 1, proc.stdout
    assert "문항별 퇴행: 1건 (q_0025)" in proc.stdout
    assert "FAIL" in proc.stdout.split("error_rate(")[1].splitlines()[0]


def test_r01_evolution_counts_budget_abstention_as_failure(tmp_path):
    from adapter import RagDomain
    from rrsi.domain import EvalSummary
    d = RagDomain()
    tid = next(i for i, it in d.items.items() if not it["answerable"])
    for job, reason in (("good", "insufficient_evidence"), ("late", "deadline_exceeded")):
        out = tmp_path / "jobs" / job / "s0"
        out.mkdir(parents=True)
        (out / f"{tid}.json").write_text(json.dumps(
            {"abstained": True, "reason": reason, "error": None, "usage": {"total_tokens": 1}}),
            encoding="utf-8")
    per_g, ex_g = d.score(tmp_path, "good", [tid], 1)
    per_l, ex_l = d.score(tmp_path, "late", [tid], 1)
    assert per_g[tid].rewards == [1.0] and per_l[tid].rewards == [0.0]
    assert ex_g["error_rate"] == 0.0 and ex_l["error_rate"] == 1.0
    assert ex_l["abstain_unanswerable"] == 0.0
    v = d.guards(EvalSummary(S=1.0, C=1.0, k=1, extra=ex_g), EvalSummary(S=0.0, C=1.0, k=1, extra=ex_l))
    assert any("error_rate" in x for x in v)


# ---------------------------------------------------------------------------
# R02 문항별 결과 검증
# ---------------------------------------------------------------------------
def _base() -> dict:
    return json.loads(BASE.read_text(encoding="utf-8"))


def _mut(fn):
    rep = _base()
    fn(rep)
    return rep


@pytest.mark.parametrize("name, fn, msg", [
    ("items 누락", lambda r: r.pop("items"), "items"),
    ("빈 배열", lambda r: r.update(items=[]), "items"),
    ("중복", lambda r: r["items"].__setitem__(1, copy.deepcopy(r["items"][0])), "중복"),
    ("개수 부족", lambda r: r["items"].pop(), "items 개수"),
    ("ID 불일치", lambda r: r["items"][0].update(id="q_9999"), "ID 집합"),
    ("순서 불일치", lambda r: r["items"].reverse(), "순서"),
    ("잘못된 타입 correct", lambda r: r["items"][0].update(correct="1"), "correct"),
    ("잘못된 타입 abstained", lambda r: r["items"][0].update(abstained="false"), "abstained"),
    ("허용 밖 outcome", lambda r: r["items"][0].update(outcome="great"), "outcome"),
    ("outcome·reason 불일치", lambda r: r["items"][-1].update(reason="deadline_exceeded"), "outcome"),
    ("error·outcome 불일치", lambda r: r["items"][-1].update(error="boom"), "outcome"),
    ("reward 범위", lambda r: r["items"][0].update(reward=1.5), "reward"),
    ("reward·correct 불일치", lambda r: r["items"][0].update(correct=0.0), "reward"),
    ("집계 S 불일치", lambda r: r.update(S=1.0), "S="),
    ("집계 correct 불일치", lambda r: r.update(correct=1.0), "correct="),
    ("outcomes 불일치", lambda r: r["outcomes"].update(abstained=0, answered=12), "outcomes"),
    ("n_unique 불일치", lambda r: r.update(n_unique=11), "n_unique"),
    ("n_unique 누락", lambda r: r.pop("n_unique"), "n_unique"),
])
def test_r02_bad_items_are_rejected(tmp_path, name, fn, msg):
    rep = _mut(fn)
    for args in ((), ("--item-regressions",)):
        proc = _check(rep, tmp_path, *args)
        assert proc.returncode == 1, (name, proc.stdout)
        assert msg in proc.stdout, (name, proc.stdout)
    # 기준선으로 등록할 때도 같은 검증을 거칩니다.
    nb = tmp_path / "new_base.json"
    proc = _check(rep, tmp_path, "--init-baseline", base=nb)
    assert proc.returncode == 1 and not nb.exists(), name


def test_r02_baseline_without_items_is_rejected(tmp_path):
    """검토자 재현: 기준선에서 items만 지우고 정상 보고서와 --item-regressions로 비교."""
    bad = tmp_path / "base_no_items.json"
    bad.write_text(json.dumps(_mut(lambda r: r.pop("items")), ensure_ascii=False), encoding="utf-8")
    proc = _check(_base(), tmp_path, "--item-regressions", base=bad)
    assert proc.returncode == 1 and "기준선: items" in proc.stdout


def test_r02_valid_report_still_passes(tmp_path):
    proc = _check(_base(), tmp_path, "--item-regressions")
    assert proc.returncode == 0, proc.stdout


# ---------------------------------------------------------------------------
# R03 정답 판정 한 번, 심판 장애 구분, 심판 표식
# ---------------------------------------------------------------------------
class _AltJudge:
    """같은 답변에 1.0, 0.0을 번갈아 내는 합성 심판. 호출 수를 셉니다."""
    mode, model = "remote", "alt-judge"

    def __init__(self, fail: bool = False) -> None:
        self.calls, self.fail = 0, fail

    def ask(self, state, questions):
        from harness.llm import DecisionUnavailable
        self.calls += 1
        if self.fail:
            raise DecisionUnavailable("judge down")
        return {"same": {"noul": 1.0 if self.calls % 2 else 0.0}}


def test_r03_judge_called_once_and_reward_matches_correct(env):
    j = _AltJudge()
    rep = rs.evaluate(env["ids"], env["items"], env["texts"], env["pipe"], j)
    judged = [r for r in rep["items"] if r["judge"] == "ok"]
    assert j.calls == len(judged) == 11                      # 답변 불가 문항은 부르지 않음
    for r in judged:
        # 인용 정밀도 1.0인 답변: 정답이면 1.0, 오답이면 0.35. reward와 correct가 같은 판정입니다.
        assert r["reward"] == (1.0 if r["correct"] == 1.0 else 0.35), r
    assert rep["judge"] == "remote/alt-judge@-/same-v1"


def test_r03_alternating_judge_report_passes_consistency_but_not_comparable(env, tmp_path):
    rep = rs.evaluate(env["ids"], env["items"], env["texts"], env["pipe"], _AltJudge())
    proc = _check(rep, tmp_path)
    assert proc.returncode == 1 and "비교할 수 없음: judge" in proc.stdout


def test_r03_judge_failure_is_not_a_wrong_answer(env, tmp_path):
    j = _AltJudge(fail=True)
    rep = rs.evaluate(env["ids"], env["items"], env["texts"], env["pipe"], j)
    rows = [r for r in rep["items"] if r["answerable"]]
    assert all(r["judge"] == "error" and r["correct"] is None for r in rows)
    assert rep["judge_errors"] == 11 and rep["correct"] is None
    nb = tmp_path / "nb.json"
    proc = _check({**rep, "correct": 0.0}, tmp_path, "--init-baseline", base=nb)
    assert proc.returncode == 1 and not nb.exists()
    assert "심판 장애" in proc.stdout or "correct" in proc.stdout


def test_r03_failure_outcomes_skip_judge(env):
    from judge import score_trial
    calls = []
    item = env["items"]["q_0001"]
    for tr in ({"abstained": True, "reason": "deadline_exceeded", "error": None, "answer": "x"},
               {"abstained": True, "reason": "x", "error": "boom", "answer": "x"},
               {"abstained": True, "reason": "insufficient_evidence", "error": None}):
        sc = score_trial(item, tr, env["texts"], lambda a: calls.append(a) or (1.0, "ok"))
        assert sc["judge"] == "skipped" and sc["correct"] == 0.0
    assert calls == []
    assert FAILURES >= {"budget_exceeded", "ops_failure", "unknown", "error"}


def test_r03_adapter_judges_once_and_counts_judge_errors(tmp_path):
    from adapter import RagDomain
    from rrsi.domain import EvalSummary
    d = RagDomain()
    tid = next(i for i, it in d.items.items() if it["answerable"])
    out = tmp_path / "jobs" / "j" / "s0"
    out.mkdir(parents=True)
    (out / f"{tid}.json").write_text(json.dumps(
        {"abstained": False, "reason": "", "error": None, "answer": "답", "citations": [],
         "retrieved": [], "usage": {"total_tokens": 1}}), encoding="utf-8")
    d._judge, d._judge_cache = _AltJudge(), {}
    d.score(tmp_path, "j", [tid], 1)
    assert d._judge.calls == 1
    d._judge, d._judge_cache = _AltJudge(fail=True), {}
    _, extra = d.score(tmp_path, "j", [tid], 1)
    assert extra["judge_errors"] == 1 and extra["judge"] == "remote/alt-judge@-/same-v1"
    v = d.guards(EvalSummary(S=1.0, C=1.0, k=1, extra={**extra, "judge_errors": 0}),
                 EvalSummary(S=1.0, C=1.0, k=1, extra=extra))
    assert any("judge errors" in x for x in v)
