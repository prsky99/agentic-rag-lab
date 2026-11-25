"""5차 검토 회귀 시험: 진화 기준선 비교 조건(F01), δ 보정 지문(F02), 라운드 ID(F05),
복사 모드 가상환경 제외(F06), 문항별 수치 경계(F07), 심판 표식(F08)."""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rrsi import loop
from rrsi.domain import Domain, EvalSummary, TaskResult
from rrsi.gitops import Workspace
from rrsi.history import History
from rrsi.selection import comparability, decide

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "eval" / "check_thresholds.py"
BASE = ROOT / "eval" / "reports" / "base_ragas.json"
UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}
CFG = {"beta0": 0.03, "beta1": 10.0, "w_s": 0.0, "w_c": 12.0, "w_n": 0.5,
       "invalid_missing_frac": 0.1}


# ---------------------------------------------------------------------------
# F01 기준선 비교 조건
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rag():
    from adapter import RagDomain
    return RagDomain()


def _extra(rag) -> dict:
    return {"cite_precision": 1.0, "abstain_answerable": 0.0, "abstain_unanswerable": 1.0,
            "p95_ms": 10.0, "goldenset_version": rag.goldenset_version, "error_rate": 0.0,
            "evaluator": rag.EVALUATOR_VERSION, "judge": "local/local-fallback/same-v1",
            "judge_errors": 0}


def _pair(rag, inc_extra=None, cand_extra=None, inc_cond=None, cand_cond=None):
    cond = {"ids": ["q_0001"], "k": 2, "evaluator": rag.EVALUATOR_VERSION,
            "judge": "local/local-fallback/same-v1",
            "goldenset": rag.goldenset_version, "runtime": {"decision": {"mode": "local"}}}
    full = {"n": 1, "missing_frac": 0.0, "trials_expected": 2, "trials_done": 2, "cost_trials": 2}
    inc = EvalSummary(S=0.8, C=100.0, k=2, extra=inc_extra or _extra(rag),
                      conditions=inc_cond or cond, **full)
    cand = EvalSummary(S=0.85, C=100.0, k=2, extra=cand_extra or _extra(rag),
                       conditions=cand_cond or copy.deepcopy(cond), **full)
    return inc, cand


def _decide(rag, inc, cand):
    return decide(cand, inc, 0.011, CFG, rag.guards(inc, cand), 0.0, s_star=inc.S)


def test_f01_same_conditions_are_compared(rag):
    inc, cand = _pair(rag)
    assert rag.guards(inc, cand) == [] and comparability(inc, cand) == []
    assert _decide(rag, inc, cand).accepted


@pytest.mark.parametrize("side,key,value", [
    ("cand", "evaluator", "rag-eval/4"),                     # 이전 평가기
    ("cand", "judge", "remote/other@ep-1/same-v1"),          # 다른 심판
    ("cand", "goldenset_version", "goldenset_v0@sha256:0"),  # 다른 골든셋
    ("inc", "judge_errors", 1),                              # 기준선 심판 장애
    ("cand", "judge_errors", 1),
    ("inc", "error_rate", None),                             # 기준선 extra 누락
    ("cand", "cite_precision", None),                        # 후보 extra 누락
    ("cand", "p95_ms", float("nan")),                        # 비유한 값
])
def test_f01_incomparable_or_invalid_extra_stops_promotion(rag, side, key, value):
    ex = _extra(rag)
    if value is None:
        del ex[key]
    else:
        ex[key] = value
    inc, cand = _pair(rag, **({"inc_extra": ex} if side == "inc" else {"cand_extra": ex}))
    assert rag.guards(inc, cand), (side, key)
    assert not _decide(rag, inc, cand).accepted


def test_f01_empty_incumbent_extra_is_not_a_free_pass(rag):
    inc, cand = _pair(rag)
    inc.extra = {}
    assert any("extra missing" in g for g in rag.guards(inc, cand))
    assert not _decide(rag, inc, cand).accepted


@pytest.mark.parametrize("change", ["ids", "k", "runtime", "missing"])
def test_f01_condition_mismatch_is_rejected_by_decide(rag, change):
    inc, cand = _pair(rag)
    if change == "ids":
        cand.conditions["ids"] = ["q_0002"]
    elif change == "k":
        cand.conditions["k"], cand.k = 3, 3
    elif change == "runtime":
        cand.conditions["runtime"] = {"decision": {"mode": "remote"}}
    else:
        inc.conditions = {}
    d = decide(cand, inc, 0.011, CFG, [], 0.0, s_star=inc.S)   # 가드가 비어 있어도 멈춥니다
    assert not d.accepted and d.reason.startswith("incomparable")


def test_f01_candidate_harness_values_are_not_conditions(rag, tmp_path, monkeypatch):
    """후보가 바꿀 수 있는 config.yaml 값은 비교 조건에 없고, 실행 모드는 있습니다."""
    import shutil
    root = tmp_path / "r"
    (root / "domains").mkdir(parents=True)
    shutil.copytree(ROOT / "domains" / "rag", root / "domains" / "rag",
                    ignore=shutil.ignore_patterns("__pycache__"))
    ids = rag.evolve_ids()[:2]
    a = rag.conditions(root, ids, 2)
    cfg = root / "domains" / "rag" / "harness" / "config.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("top_k: 40", "top_k: 30"),
                   encoding="utf-8")
    assert rag.conditions(root, ids, 2) == a
    monkeypatch.setenv("DECISION_MODE", "offline")
    assert rag.conditions(root, ids, 2)["runtime"] != a["runtime"]
    assert "config" not in json.dumps(a["runtime"])


class _FakeDomain(Domain):
    """점수·조건·심판 장애를 바꿀 수 있는 합성 도메인(외부 호출 없음)."""
    name = "fake"

    def __init__(self, S=0.8, cond="c1", judge_errors=0):
        self.S, self.cond, self.judge_errors = S, cond, judge_errors

    def evolve_ids(self):
        return ["a", "b"]

    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        return None

    def score(self, runs_dir, job, ids, k):
        per = {i: TaskResult(rewards=[self.S] * k, tokens=[10] * k) for i in ids}
        return per, {"judge_errors": self.judge_errors}

    def conditions(self, root, ids, k):
        return {"ids": sorted(ids), "k": k, "env": self.cond}

    def validity(self, summary):
        je = summary.extra.get("judge_errors")
        return [f"judge errors {je}"] if je else []


def _fake_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "domains" / "fake" / "harness").mkdir(parents=True)
    (root / "domains" / "fake" / "harness" / "h.txt").write_text("v1", encoding="utf-8")
    return root, root / "runs" / "fake"


def test_f01_baseline_with_judge_errors_is_not_registered(tmp_path):
    root, runs = _fake_root(tmp_path)
    with pytest.raises(loop.BaselineInvalid):
        loop.baseline(_FakeDomain(judge_errors=1), root, runs, {"k": 2}, "base")
    assert not (runs / "frontier.json").exists()


def test_f01_resume_resets_frontier_when_conditions_or_harness_change(tmp_path):
    root, runs = _fake_root(tmp_path)
    fr = loop.baseline(_FakeDomain(S=0.9), root, runs, {"k": 2}, "base")
    assert fr["incumbent"]["conditions"]["env"] == "c1" and fr["S_star"] == 0.9
    # 같은 조건이면 그대로 이어 씁니다(T=0: 라운드 없이 재개 검사만)
    fr = loop.run_rounds(_FakeDomain(S=0.5), root, runs, {"k": 2}, 0)
    assert fr["S_star"] == 0.9 and not list(runs.glob("frontier.stale-*.json"))
    # 조건이 바뀌면 이전 S*를 버리고 새 기준선으로 재설정합니다
    fr = loop.run_rounds(_FakeDomain(S=0.5, cond="c2"), root, runs, {"k": 2}, 0)
    assert fr["S_star"] == 0.5 and fr["incumbent"]["conditions"]["env"] == "c2"
    assert len(list(runs.glob("frontier.stale-*.json"))) == 1
    # 현행 하네스를 직접 고쳐도 재설정합니다
    (root / "domains" / "fake" / "harness" / "h.txt").write_text("v2", encoding="utf-8")
    fr2 = loop.run_rounds(_FakeDomain(S=0.6, cond="c2"), root, runs, {"k": 2}, 0)
    assert fr2["S_star"] == 0.6 and fr2["incumbent"]["harness_tree"] != fr["incumbent"]["harness_tree"]


def test_f01_resume_stops_when_new_baseline_has_judge_errors(tmp_path):
    root, runs = _fake_root(tmp_path)
    loop.baseline(_FakeDomain(), root, runs, {"k": 2}, "base")
    with pytest.raises(loop.BaselineInvalid):
        loop.run_rounds(_FakeDomain(cond="c2", judge_errors=1), root, runs, {"k": 2}, 0)


# ---------------------------------------------------------------------------
# F02 δ 보정 입력과 재사용 조건
# ---------------------------------------------------------------------------
FAKECAL = '''
from rrsi.domain import Domain


class D(Domain):
    name = "fakecal"

    def validity(self, s):
        je = s.extra.get("judge_errors")
        return [f"judge errors {je}"] if je else []


DOMAIN = D()
'''


def _cal_root(tmp_path: Path) -> Path:
    root = tmp_path / "cal"
    pkg = root / "domains" / "fakecal"
    pkg.mkdir(parents=True)
    (root / "domains" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "adapter.py").write_text(FAKECAL, encoding="utf-8")
    return root


def _job(root: Path, job: str, S, evaluator="rag-eval/5", judge_errors=0, **kw) -> None:
    d = root / "runs" / "fakecal" / "jobs" / job
    d.mkdir(parents=True, exist_ok=True)
    rec = {"job": job, "S": S, "C": 10.0, "k": 2, "n": 2, "missing_frac": 0.0,
           "extra": {"judge_errors": judge_errors, "evaluator": evaluator},
           "conditions": {"ids": ["a", "b"], "k": 2, "evaluator": evaluator},
           "harness_tree": "h1"}
    rec.update(kw)
    (d / "eval.json").write_text(json.dumps(rec), encoding="utf-8")


def _calibrate(root: Path, jobs: str):
    return subprocess.run([sys.executable, str(ROOT / "rrsi.py"), "--domain", "fakecal", "--root",
                           str(root), "calibrate", "--jobs", jobs],
                          cwd=str(root), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env={**UTF8, "PYTHONPATH": str(ROOT)})


def test_f02_mixed_evaluator_and_judge_errors_cannot_calibrate(tmp_path):
    root = _cal_root(tmp_path)
    _job(root, "b1", 0.9)
    _job(root, "b2", 0.5, evaluator="rag-eval/4")
    _job(root, "b3", 0.8, judge_errors=1)
    proc = _calibrate(root, "b1,b2,b3")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    assert not (root / "runs" / "fakecal" / "calibration.json").exists()


@pytest.mark.parametrize("bad", [
    {"S": float("nan")}, {"S": 10 ** 400}, {"S": True}, {"missing_frac": 0.5},
    {"conditions": {}}, {"harness_tree": ""}])
def test_f02_bad_job_is_excluded_and_two_valid_jobs_fail(tmp_path, bad):
    root = _cal_root(tmp_path)
    _job(root, "b1", 0.9)
    _job(root, "b2", 0.8)
    bad = dict(bad)
    _job(root, "b3", bad.pop("S", 0.7), **bad)
    proc = _calibrate(root, "b1,b2,b3")
    assert proc.returncode == 1 and "b3 제외" in proc.stdout, proc.stdout + proc.stderr


def test_f02_mixed_harness_cannot_calibrate(tmp_path):
    root = _cal_root(tmp_path)
    for i, h in enumerate(["h1", "h1", "h2", "h2"]):
        _job(root, f"b{i}", 0.9 - i * 0.01, harness_tree=h)
    assert _calibrate(root, "b0,b1,b2,b3").returncode == 1


def test_f02_same_conditions_calibrate_and_record_fingerprint(tmp_path):
    root = _cal_root(tmp_path)
    for i, s in enumerate([0.90, 0.91, 0.92]):
        _job(root, f"b{i}", s)
    proc = _calibrate(root, "b0,b1,b2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    runs = root / "runs" / "fakecal"
    cal = json.loads((runs / "calibration.json").read_text(encoding="utf-8"))
    assert cal["n"] == 3 and cal["delta"] > 0
    fp = cal["fingerprint"]
    assert fp["harness_tree"] == "h1" and fp["conditions"]["evaluator"] == "rag-eval/5"
    # 같은 조건이면 재사용, 조건·기준선 하네스가 바뀌면 사전값으로 돌아가고 재측정을 요청합니다
    cfg = {"delta_prior": 0.011}
    assert loop.effective_delta(runs, cfg, fp["conditions"], "h1") == (cal["delta"], "calibrated")
    other = {**fp["conditions"], "evaluator": "rag-eval/6"}
    d, src = loop.effective_delta(runs, cfg, other, "h1")
    assert d == 0.011 and "stale" in src
    d, src = loop.effective_delta(runs, cfg, fp["conditions"], "h9")
    assert d == 0.011 and "stale" in src


def test_f02_calibration_without_fingerprint_is_not_reused(tmp_path):
    (tmp_path / "calibration.json").write_text(
        json.dumps({"delta": 0.5, "n": 3, "z": 2.0, "sd_null": 0.25}), encoding="utf-8")
    d, src = loop.effective_delta(tmp_path, {"delta_prior": 0.011}, {"k": 2}, "h1")
    assert d == 0.011 and "stale" in src


# ---------------------------------------------------------------------------
# F05 재실행 시 라운드 ID
# ---------------------------------------------------------------------------
def test_f05_last_rounds_separates_runs(tmp_path):
    hist = History(tmp_path / "h.jsonl")
    for run in ("run1", "run2"):
        for t in (0, 1):
            for lab in ("A", "B"):
                hist.append({"t": t, "run": run, "label": f"r{t}{lab}", "edits": [f"{run}-{t}{lab}"]})
    last = hist.last_rounds(1)
    assert [r["label"] for r in last] == ["r1A", "r1B"] and {r["run"] for r in last} == {"run2"}
    assert [e["run"] for e in hist.recent_edits(rounds=1)] == ["run2", "run2"]
    assert len(hist.last_rounds(3)) == 6


def test_f05_round_numbers_continue_across_runs(tmp_path):
    root, runs = _fake_root(tmp_path)
    dom = _FakeDomain()
    loop.baseline(dom, root, runs, {"k": 2}, "base")
    fr = loop.run_rounds(dom, root, runs, {"k": 2}, 2)
    assert fr["rounds_done"] == 2
    fr = loop.run_rounds(dom, root, runs, {"k": 2}, 1)
    assert fr["rounds_done"] == 3


# ---------------------------------------------------------------------------
# F06 복사 모드 가상환경 제외
# ---------------------------------------------------------------------------
def test_f06_copy_mode_skips_virtualenvs_and_caches(tmp_path):
    root = tmp_path / "zip"
    (root / "domains" / "rag" / "harness").mkdir(parents=True)
    (root / "domains" / "rag" / "harness" / "x.py").write_text("print('ok')\n", encoding="utf-8")
    for d in (".venv", "venv", "myenv", "__pycache__", ".pytest_cache", "build", "dist",
              "pkg.egg-info", ".mypy_cache"):
        (root / d).mkdir()
        (root / d / "marker").write_text("x", encoding="utf-8")
    (root / "myenv" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    ws = Workspace(root, root / "runs" / "rag", "domains/rag/harness")
    assert ws.mode == "copy"
    wt = ws.candidate("r0A")
    assert (wt / "domains" / "rag" / "harness" / "x.py").is_file()
    assert not [p for p in wt.rglob("marker")]
    proc = subprocess.run([sys.executable, "x.py"], cwd=str(wt / "domains" / "rag" / "harness"),
                          capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# F07 문항별 수치 경계
# ---------------------------------------------------------------------------
def _ct():
    sys.path.insert(0, str(ROOT / "eval"))
    import check_thresholds
    return check_thresholds


@pytest.mark.parametrize("key", ["reward", "cite_precision", "correct"])
@pytest.mark.parametrize("value", [10 ** 400, -10 ** 400, float("nan"), float("inf"), True, "0.5",
                                   [1]])
def test_f07_row_numbers_never_raise(tmp_path, key, value):
    ct = _ct()
    base = json.loads(BASE.read_text(encoding="utf-8"))
    i = next(j for j, r in enumerate(base["items"]) if r["answerable"] and r["judge"] == "ok")
    base["items"][i][key] = value
    errs = ct.validate(base, "기준선", 10)
    assert errs, (key, value)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(base, allow_nan=True), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(CHECK), str(path), "--baseline", str(BASE)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=UTF8)
    assert proc.returncode == 1 and "Traceback" not in proc.stderr, proc.stderr[-400:]


def test_f07_num01_accepts_ints_and_floats_in_range():
    ct = _ct()
    assert ct._num01(0) and ct._num01(1) and ct._num01(0.5)
    assert not ct._num01(2) and not ct._num01(10 ** 400) and not ct._num01(False)
    assert not ct._num01(math.nan)


# ---------------------------------------------------------------------------
# F08 심판 설정·표식
# ---------------------------------------------------------------------------
def test_f08_judge_follows_config_and_endpoint_changes_id(monkeypatch):
    from judge import build_judge, judge_id
    monkeypatch.setenv("DECISION_MODEL", "env-model")
    monkeypatch.delenv("DECISION_MODE", raising=False)
    a = build_judge({"judge_model": "jev-x", "judge_base_url": "http://u:secret@host-a:8000/v1?k=s"})
    b = build_judge({"judge_model": "jev-x", "judge_base_url": "http://host-b:8000/v1"})
    assert a.mode == "remote" and a.model == "jev-x"           # 설정값이 실제 심판입니다
    ia, ib = judge_id(a), judge_id(b)
    assert ia != ib and ia.startswith("remote/jev-x@ep-") and ib.startswith("remote/jev-x@ep-")
    assert "secret" not in ia and "host-a" not in ia             # 비밀값·주소 원문은 남기지 않습니다
    same = build_judge({"judge_model": "jev-x", "judge_base_url": "http://other:pw@host-a:8000/v1"})
    assert judge_id(same) == ia                                  # 사용자 정보만 다르면 같은 심판
    rev = build_judge({"judge_model": "jev-x", "judge_base_url": "http://host-a:8000/v1",
                       "judge_revision": "r2"})
    assert judge_id(rev) != ia and "+r2@" in judge_id(rev)
    # 빈 값은 환경 변수를 따릅니다
    assert build_judge({"judge_model": ""}).model == "env-model"


def test_f08_local_judge_id_is_unchanged_and_adapter_uses_config(rag):
    from judge import build_judge, judge_id
    assert judge_id(build_judge({"judge_mode": "local"})) == "local/local-fallback/same-v1"
    import adapter
    assert rag._judge.model == (adapter.CFG.get("judge_model") or rag._judge.model)
    assert rag.conditions(ROOT, ["q_0001"], 1)["judge"] == judge_id(rag._judge)
