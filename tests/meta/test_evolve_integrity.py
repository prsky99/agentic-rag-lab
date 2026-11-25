"""6차 검토 회귀 시험: 기준선 산출물 완전성(G02), git 작업장 출발점(G03), 저장 값 범위·δ 유효성(G04).

외부 모델·네트워크는 쓰지 않습니다. G02의 첫 시험만 실제 RagDomain 채점을 쓰고(실행 함수만 무출력으로
바꿉니다), 나머지는 합성 도메인과 임시 git 저장소를 씁니다.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rrsi import loop
from rrsi.calibrate import job_problems
from rrsi.domain import Domain, EvalSummary, TaskResult
from rrsi.evaluate import summary_problems
from rrsi.gitops import Workspace, WorkspaceError, copy_ignore
from rrsi.selection import decide

ROOT = Path(__file__).resolve().parents[2]
CFG = {"beta0": 0.03, "beta1": 10.0, "w_s": 0.0, "w_c": 12.0, "w_n": 0.5,
       "invalid_missing_frac": 0.1}


# ---------------------------------------------------------------------------
# G02 기준선 산출물 완전성
# ---------------------------------------------------------------------------
class _Dom(Domain):
    """트라이얼 누락·비용 미측정을 만들 수 있는 합성 도메인."""
    name = "fake"

    def __init__(self, S=0.8, ids=("a", "b"), drop=0, tokens=10, cond="c1"):
        self.S, self.ids, self.drop, self.tokens, self.cond = S, list(ids), drop, tokens, cond

    def evolve_ids(self):
        return list(self.ids)

    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        return None

    def score(self, runs_dir, job, ids, k):
        per, left = {}, self.drop
        for i in ids:
            tr = TaskResult()
            for _ in range(k):
                if left > 0:                       # 누락 트라이얼: 0점, 비용 미측정
                    tr.rewards.append(0.0)
                    tr.tokens.append(None)
                    tr.missing += 1
                    left -= 1
                else:
                    tr.rewards.append(self.S)
                    tr.tokens.append(self.tokens)
            per[i] = tr
        return per, {"judge_errors": 0}

    def conditions(self, root, ids, k):
        return {"ids": sorted(ids), "k": k, "env": self.cond}


def _root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "domains" / "fake" / "harness").mkdir(parents=True)
    (root / "domains" / "fake" / "harness" / "h.txt").write_text("v1", encoding="utf-8")
    return root, root / "runs" / "fake"


def test_g02_real_rag_all_outputs_missing_is_not_registered(tmp_path, monkeypatch):
    """검토 재현: RagDomain 실행만 무출력 → 26문항×2회 52/52 누락. 기준선으로 등록하지 않습니다."""
    from adapter import RagDomain
    root = tmp_path / "zip"
    shutil.copytree(ROOT / "domains" / "rag", root / "domains" / "rag", ignore=copy_ignore)
    dom = RagDomain()
    monkeypatch.setattr(RagDomain, "run", lambda self, *a, **kw: None)
    runs = root / "runs" / "rag"
    with pytest.raises(loop.BaselineInvalid) as exc:
        loop.baseline(dom, root, runs, {"k": 2}, "base")
    msg = str(exc.value)
    assert "산출물 0/52" in msg and "missing_frac=1.0" in msg and "비용 측정 0/52" in msg
    assert not (runs / "frontier.json").exists()
    ev = json.loads((runs / "jobs" / "base" / "eval.json").read_text(encoding="utf-8"))
    assert ev["trials_expected"] == 52 and ev["trials_done"] == 0 and ev["cost_trials"] == 0
    # 분모 0인 비율은 정상값(기권 100%·오류 0)이 아니라 미측정(None)입니다
    assert ev["extra"]["abstain_unanswerable"] is None and ev["extra"]["error_rate"] is None
    assert any("error_rate unmeasured" in x for x in dom.validity(EvalSummary(
        S=0.0, C=0.0, k=2, extra=ev["extra"], conditions=ev["conditions"])))


@pytest.mark.parametrize("dom, why", [
    (_Dom(drop=1), "산출물 3/4"),                      # 부분 누락
    (_Dom(drop=4), "산출물 0/4"),                      # 전부 누락
    (_Dom(ids=()), "n=0"),                             # 빈 기준선(문항 0)
    (_Dom(tokens=None), "비용 측정 0/4"),              # 산출물은 있으나 비용 미측정
])
def test_g02_incomplete_or_empty_baseline_is_rejected(tmp_path, dom, why):
    root, runs = _root(tmp_path)
    with pytest.raises(loop.BaselineInvalid) as exc:
        loop.baseline(dom, root, runs, {"k": 2}, "base")
    assert why in str(exc.value)
    assert not (runs / "frontier.json").exists()


def test_g02_complete_baseline_keeps_counts_and_zero_tokens_is_real_zero(tmp_path):
    root, runs = _root(tmp_path)
    fr = loop.baseline(_Dom(tokens=0), root, runs, {"k": 2}, "base")   # 모든 트라이얼을 재고 0토큰
    inc = fr["incumbent"]
    assert inc["C"] == 0.0 and inc["n"] == 2 and inc["missing_frac"] == 0.0
    assert (inc["trials_expected"], inc["trials_done"], inc["cost_trials"]) == (4, 4, 4)
    assert loop.resume_problems(_Dom(), root, Workspace(root, runs, "domains/fake/harness"),
                                fr, {"k": 2}, ["a", "b"]) == []


@pytest.mark.parametrize("patch", [
    {"missing_frac": 1.0, "trials_done": 0, "cost_trials": 0, "S": 0.0, "C": 0.0},
    {"trials_expected": None, "trials_done": None, "cost_trials": None, "n": None},  # 이전 판
    {"n": 1},
])
def test_g02_bad_saved_frontier_is_not_resumed(tmp_path, patch):
    root, runs = _root(tmp_path)
    fr = loop.baseline(_Dom(S=0.9), root, runs, {"k": 2}, "base")
    for kk, val in patch.items():
        if val is None:
            fr["incumbent"].pop(kk)
        else:
            fr["incumbent"][kk] = val
    (runs / "frontier.json").write_text(json.dumps(fr), encoding="utf-8")
    ws = Workspace(root, runs, "domains/fake/harness")
    assert loop.resume_problems(_Dom(), root, ws, fr, {"k": 2}, ["a", "b"])
    # 재개 대신 보관 후 새로 잽니다. 새 기준선도 불완전하면 멈춥니다.
    with pytest.raises(loop.BaselineInvalid):
        loop.run_rounds(_Dom(drop=4), root, runs, {"k": 2}, 0)
    out = loop.run_rounds(_Dom(S=0.7), root, runs, {"k": 2}, 0)
    assert out["incumbent"]["trials_done"] == 4 and out["S_star"] == 0.7
    assert len(list(runs.glob("frontier.stale-*.json"))) == 2


def test_g02_candidate_with_missing_trials_is_not_promoted():
    full = dict(n=1, missing_frac=0.0, trials_expected=2, trials_done=2, cost_trials=2)
    inc = EvalSummary(S=0.5, C=100.0, k=2, conditions={"k": 2}, **full)
    part = dict(full, missing_frac=0.05, trials_done=1, cost_trials=1)
    cand = EvalSummary(S=0.9, C=10.0, k=2, conditions={"k": 2}, **part)
    d = decide(cand, inc, 0.011, CFG, [], 0.0, s_star=0.5)
    assert not d.accepted and d.reason.startswith("incomplete")
    ok = EvalSummary(S=0.9, C=100.0, k=2, conditions={"k": 2}, **full)
    assert decide(ok, inc, 0.011, CFG, [], 0.0, s_star=0.5).accepted


# ---------------------------------------------------------------------------
# G03 git 모드 출발점
# ---------------------------------------------------------------------------
def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=check)


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git 없음")
    root, runs = _root(tmp_path)
    (root / ".gitignore").write_text("runs/\n*.cache\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root, runs


def test_g03_clean_clone_makes_candidate_from_baseline(repo):
    root, runs = repo
    fr = loop.baseline(_Dom(), root, runs, {"k": 2}, "base")
    ws = Workspace(root, runs, "domains/fake/harness")
    assert ws.mode == "git"
    wt = ws.candidate("r0A")
    assert Workspace(wt, runs, ws.harness_rel).tree_hash() == fr["incumbent"]["harness_tree"]
    ws.cleanup(wt)
    out = loop.run_rounds(_Dom(), root, runs, {"k": 2}, 1)   # 편집 없는 라운드 1회
    assert out["rounds_done"] == 1


@pytest.mark.parametrize("kind", ["unstaged", "staged", "untracked"])
def test_g03_uncommitted_changes_stop_baseline_and_run(repo, kind):
    root, runs = repo
    loop.baseline(_Dom(), root, runs, {"k": 2}, "base")
    h = root / "domains" / "fake" / "harness"
    if kind == "unstaged":
        (h / "h.txt").write_text("dirty", encoding="utf-8")
    elif kind == "staged":
        (h / "h.txt").write_text("dirty", encoding="utf-8")
        _git(root, "add", "-A")
    else:
        (h / "new.txt").write_text("x", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="커밋하지 않은 변경"):
        loop.baseline(_Dom(), root, runs, {"k": 2}, "base")
    with pytest.raises(WorkspaceError, match="커밋하지 않은 변경"):
        loop.run_rounds(_Dom(), root, runs, {"k": 2}, 1)
    assert (h / ("new.txt" if kind == "untracked" else "h.txt")).exists()   # 사용자 변경은 그대로


@pytest.mark.parametrize("kind", ["diverged", "behind"])
def test_g03_evolve_branch_mismatch_stops(repo, kind):
    root, runs = repo
    h = root / "domains" / "fake" / "harness" / "h.txt"
    _git(root, "branch", "evolve/rag")
    if kind == "diverged":
        _git(root, "checkout", "-q", "evolve/rag")
        h.write_text("v-evolve", encoding="utf-8")
        _git(root, "commit", "-q", "-am", "evolve side")
        _git(root, "checkout", "-q", "main")
        h.write_text("v-main", encoding="utf-8")
        _git(root, "commit", "-q", "-am", "main side")
    else:
        h.write_text("v2", encoding="utf-8")
        _git(root, "commit", "-q", "-am", "v2")
        _git(root, "branch", "-f", "evolve/rag", "HEAD")
        _git(root, "reset", "-q", "--hard", "HEAD~1")             # HEAD가 evolve/rag보다 뒤
    with pytest.raises(WorkspaceError, match="분기 또는 역행"):
        loop.baseline(_Dom(), root, runs, {"k": 2}, "base")
    assert not (runs / "frontier.json").exists()


def test_g03_start_harness_must_match_baseline(repo):
    """git이 무시하는 파일이 하네스에 있으면 후보 출발 하네스가 기준선과 달라 멈춥니다."""
    root, runs = repo
    loop.baseline(_Dom(), root, runs, {"k": 2}, "base")
    (root / "domains" / "fake" / "harness" / "local.cache").write_text("x", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="후보 출발 하네스"):
        # 재개 검사는 하네스 변경으로 기준선을 다시 재고, 후보는 그 기준선과 달라 멈춥니다
        loop.run_rounds(_Dom(), root, runs, {"k": 2}, 1)
    assert not (runs / "wt" / "r0A").exists()               # 멈춘 후보 작업장은 정리합니다


def test_g03_failed_promote_commit_is_not_success(repo):
    root, runs = repo
    ws = Workspace(root, runs, "domains/fake/harness")
    ws.ensure_baseline()
    wt = ws.candidate("r0A")
    (wt / "domains" / "fake" / "harness" / "h.txt").write_text("cand", encoding="utf-8")
    hook = root / ".git" / "hooks" / "pre-commit"
    hook.write_bytes(b"#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    with pytest.raises(WorkspaceError, match="승격 커밋 실패"):
        ws.promote(wt, "rrsI test")
    assert _git(root, "rev-parse", "HEAD").stdout.strip() == head
    assert (root / "domains" / "fake" / "harness" / "h.txt").read_text(encoding="utf-8") == "v1"
    assert ws.dirty_paths() == []
    hook.unlink()
    assert ws.promote(wt, "rrsI test") != _git(root, "rev-parse", "--short", head).stdout.strip()
    ws.cleanup(wt)


# ---------------------------------------------------------------------------
# G04 저장 값 범위·δ 유효성
# ---------------------------------------------------------------------------
def test_g04_out_of_range_values_fail_validity():
    from adapter import RagDomain
    rag = RagDomain()
    ex = {"cite_precision": 2, "abstain_answerable": 0.0, "abstain_unanswerable": 1.0,
          "p95_ms": -5.0, "error_rate": -1, "goldenset_version": rag.goldenset_version,
          "evaluator": rag.EVALUATOR_VERSION, "judge": "j", "judge_errors": 0}
    v = " | ".join(rag.validity(EvalSummary(S=2.0, C=-1.0, k=0, extra=ex)))
    for part in ("cite_precision=2", "error_rate=-1", "p95_ms=-5.0", "S=2.0", "C=-1.0", "k=0"):
        assert part in v, v
    s = EvalSummary(S=2.0, C=-1.0, k=2, n=1, missing_frac=0.0, trials_expected=2,
                    trials_done=2, cost_trials=2)
    v = " | ".join(summary_problems(s, ["a"], 2))
    assert "S=2.0" in v and "C=-1.0" in v
    assert summary_problems(EvalSummary(S=float("nan"), C=float("inf"), k=True), ["a"], 2)
    bad = {"S": 0.5, "C": -1.0, "k": 0, "n": 2, "missing_frac": 0.0, "conditions": {"k": 2},
           "harness_tree": "h"}
    jp = " | ".join(job_problems(bad))
    assert "C=-1.0" in jp and "k=0" in jp


def _calib(tmp_path: Path, **kw) -> Path:
    rec = {"delta": 0.04, "sd_null": 0.02, "sigma": 0.01414, "z": 2.0, "n": 3,
           "method": "sqrt2-sigma", "fingerprint": {"harness_tree": "h1", "conditions": {"k": 2}}}
    rec.update(kw)
    (tmp_path / "calibration.json").write_text(json.dumps(rec), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("kw", [
    {"delta": float("inf")}, {"delta": 5.0, "sd_null": 2.5}, {"delta": 0.9},
    {"z": float("nan")}, {"n": 2}, {"n": True}, {"fingerprint": {"harness_tree": 1, "conditions": {}}}])
def test_g04_abnormal_calibration_falls_back_to_prior(tmp_path, kw):
    runs = _calib(tmp_path, **kw)
    d, src = loop.effective_delta(runs, {"delta_prior": 0.011}, {"k": 2}, "h1")
    assert d == 0.011 and "calibration invalid" in src


def test_g04_valid_calibration_is_still_used(tmp_path):
    runs = _calib(tmp_path)
    assert loop.effective_delta(runs, {"delta_prior": 0.011}, {"k": 2}, "h1") == (0.04, "calibrated")
