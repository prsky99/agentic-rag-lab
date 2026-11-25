"""회귀 평가 신뢰성 시험(3차 검토 E05·E06).

* E05: 답변 불가 문항 q_0025의 검색기가 TimeoutError를 내면 파이프라인은 통제된 기권
  (abstained=true, reason=retrieval_unavailable, error 있음)으로 끝납니다. 이 결과를 회귀 평가가
  "정상 기권 만점"으로 세어 기준선과 같은 점수로 통과하던 문제를 재현합니다.
* E06: 기준선과 다른 문항, 중복 문항, 없는 문항으로 만든 보고서가 비교를 통과하던 문제와
  집계 지표의 상쇄(한 문항이 새로 틀리고 다른 문항이 새로 맞음)를 재현합니다.
"""
from __future__ import annotations

import copy
import json
import os
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
    items = rs.load_goldenset()
    return {"items": items, "texts": rs.chunk_texts(), "judge": DecisionClient(),
            "pipe": RAGPipeline.from_config(ROOT / "domains/rag/harness/config.yaml"),
            "ids": rs.load_subset(ROOT / "eval/ci/regression_subset.json")}


class _TimeoutFor:
    """지정한 질문에서만 TimeoutError를 내는 검색기(나머지는 실제 검색기에 넘깁니다)."""

    def __init__(self, inner, question: str) -> None:
        self.inner, self.question = inner, question

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def search(self, question, tenant, top_k=40):
        if question == self.question:
            raise TimeoutError("timed out")
        return self.inner.search(question, tenant, top_k=top_k)


# ---------------------------------------------------------------------------
# E05 오류 기권은 정상 기권 만점이 아님
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def timeout_report(env):
    pipe = env["pipe"]
    real = pipe.retriever
    pipe.retriever = _TimeoutFor(real, env["items"]["q_0025"]["question"])
    try:
        return rs.evaluate(env["ids"], env["items"], env["texts"], pipe, env["judge"],
                           subset="eval/ci/regression_subset.json")
    finally:
        pipe.retriever = real


def test_e05_retrieval_timeout_is_error_not_abstention(timeout_report):
    row = next(r for r in timeout_report["items"] if r["id"] == "q_0025")
    assert row["abstained"] is True and row["reason"] == "retrieval_unavailable"
    assert row["error"] and row["outcome"] == "error" and row["reward"] == 0.0
    assert timeout_report["abstain_unanswerable"] == 0.0
    assert timeout_report["error_rate"] == round(1 / 12, 4)
    assert timeout_report["outcomes"]["error"] == 1
    assert timeout_report["outcomes"]["answered"] == 11 and timeout_report["outcomes"]["abstained"] == 0
    base = json.loads(BASE.read_text(encoding="utf-8"))
    assert timeout_report["S"] < base["S"]


def test_e05_regression_check_fails_on_error_increase(timeout_report, tmp_path):
    proc = _check(timeout_report, tmp_path)
    assert proc.returncode == 1, proc.stdout
    assert "error_rate(실패: 오류·장애·예산): base 0.0000 → now 0.0833" in proc.stdout
    assert "FAIL" in proc.stdout.split("error_rate(실패: 오류·장애·예산)")[1].splitlines()[0]


def test_e05_regression_and_evolution_use_same_reward(env):
    from adapter import RagDomain
    from judge import answer_reward
    d = RagDomain()
    item = env["items"]["q_0025"]
    err = {"abstained": True, "reason": "retrieval_unavailable", "error": "retrieval_unavailable: x"}
    ok = {"abstained": True, "reason": "insufficient_evidence", "error": None}
    for tr in (err, ok):
        assert answer_reward(item, tr, env["texts"], env["judge"]) == d._reward(item, tr)[0]
    assert answer_reward(item, err, env["texts"], env["judge"]) == 0.0
    assert answer_reward(item, ok, env["texts"], env["judge"]) == 1.0


# ---------------------------------------------------------------------------
# E06 문항 목록 검사와 비교 가능 조건
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ids, msg", [
    (["q_0001"] * 11 + ["q_0025"], "중복"),
    ([], "비어"),
    (["q_0001", "q_9999"], "없는"),
])
def test_e06_bad_subset_is_rejected_before_evaluation(tmp_path, ids, msg):
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps({"ids": ids}), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(ROOT / "eval" / "run_subset.py"),
                           "--subset", str(subset), "--out", str(tmp_path / "r.json"),
                           "--config", str(ROOT / "domains/rag/harness/config.yaml")],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=str(ROOT), env=UTF8)
    assert proc.returncode == 2 and msg in proc.stdout
    assert not (tmp_path / "r.json").exists()


def test_e06_different_items_are_not_comparable(env, tmp_path):
    other = ["q_0002", "q_0004", "q_0006", "q_0008", "q_0010", "q_0012", "q_0014",
             "q_0017", "q_0018", "q_0026"]
    rep = rs.evaluate(other, env["items"], env["texts"], env["pipe"], env["judge"])
    assert rep["S"] == 1.0                                         # 점수만 보면 더 좋습니다
    proc = _check(rep, tmp_path)
    assert proc.returncode == 1 and "비교할 수 없음: subset_sha256" in proc.stdout


def test_e06_forged_duplicate_report_is_rejected(tmp_path):
    base = json.loads(BASE.read_text(encoding="utf-8"))
    forged = {**base, "ids": ["q_0001"] * 11 + ["q_0025"]}         # 해시는 기준선 값 그대로
    proc = _check(forged, tmp_path)
    assert proc.returncode == 1 and "중복" in proc.stdout


def test_e06_subset_hash_ignores_order():
    assert rs.subset_sha256(["q_0003", "q_0001"]) == rs.subset_sha256(["q_0001", "q_0003"])
    assert rs.subset_sha256(["q_0001"]) != rs.subset_sha256(["q_0001", "q_0003"])


def _swap(base: dict) -> dict:
    """기준선에서 맞던 문항 하나를 틀리게, 틀리던 문항 하나를 맞게 바꿉니다(집계는 그대로)."""
    rep = copy.deepcopy(base)
    rows = [r for r in rep["items"] if r["answerable"]]
    good = next(r for r in rows if r["correct"] == 1.0)
    bad = next(r for r in rows if r["correct"] != 1.0)
    good["correct"], bad["correct"] = 0.0, 1.0
    good["reward"], bad["reward"] = bad["reward"], good["reward"]   # 보상식과 맞게(R02 검사)
    return rep


def test_e06_item_regression_option_catches_offsetting_changes(tmp_path):
    base = json.loads(BASE.read_text(encoding="utf-8"))
    rep = _swap(base)
    assert rep["correct"] == base["correct"]
    assert _check(rep, tmp_path).returncode == 0                   # 집계만 보면 통과(상쇄)
    proc = _check(rep, tmp_path, "--item-regressions")
    assert proc.returncode == 1 and "문항별 퇴행: 1건" in proc.stdout


def test_e06_update_baseline_registers_new_composition(env, tmp_path):
    base = tmp_path / "base.json"
    base.write_text(BASE.read_text(encoding="utf-8"), encoding="utf-8")
    rep = rs.evaluate(env["ids"][2:], env["items"], env["texts"], env["pipe"], env["judge"])
    assert _check(rep, tmp_path, base=base).returncode == 1         # 그대로는 비교할 수 없음
    proc = _check(rep, tmp_path, "--update-baseline", base=base)
    assert proc.returncode == 0 and "subset_sha256" in proc.stdout
    assert _check(rep, tmp_path, base=base).returncode == 0
