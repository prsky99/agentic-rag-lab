"""진화·평가 체계의 신뢰성 회귀 시험(편집 경계, diff, 결과 재사용, δ, 오류 채점, 기준선, 채점 캐시)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rrsi import critic
from rrsi.calibrate import calibrate
from rrsi.propose import Edit, EditError, plan_edits

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = [(r"gold_answer", "eval artifact")]


def _repo(tmp_path: Path) -> Path:
    h = tmp_path / "domains" / "rag" / "harness"
    h.mkdir(parents=True)
    (h / "config.yaml").write_text("top_k: 8\nthreshold: 0.65\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("keep\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("file", [
    "domains/rag/harness/../../../secret.txt",   # 상위 폴더로 빠져나가기
    "/etc/passwd",                               # 절대 경로
    "domains/rag/harness_evil/config.yaml",      # 접두어만 같은 다른 폴더
])
def test_edit_outside_surface_is_rejected_before_write(tmp_path, file):
    root = _repo(tmp_path)
    with pytest.raises(EditError):
        plan_edits(root, [Edit("x", "h", file, "keep", "changed")])
    assert (root / "secret.txt").read_text(encoding="utf-8") == "keep\n"


def test_ambiguous_or_empty_anchor_is_rejected(tmp_path):
    root = _repo(tmp_path)
    f = "domains/rag/harness/config.yaml"
    (root / f).write_text("a: 1\na: 1\n", encoding="utf-8")
    with pytest.raises(EditError, match="ambiguous"):
        plan_edits(root, [Edit("x", "h", f, "a: 1", "a: 2")])
    with pytest.raises(EditError, match="empty"):
        plan_edits(root, [Edit("x", "h", f, "", "a: 2")])


def test_plan_does_not_write_and_multiline_diff_is_screened(tmp_path):
    root = _repo(tmp_path)
    f = "domains/rag/harness/config.yaml"
    diff, writes = plan_edits(root, [Edit("x", "h", f, "threshold: 0.65", "threshold: 0.6\nuse: gold_answer")])
    assert (root / f).read_text(encoding="utf-8").count("0.65") == 1     # 아직 쓰지 않음
    assert "+use: gold_answer" in diff                                   # 두 번째 줄도 + 줄
    assert not critic.screen(diff, PATTERNS).ok                           # 비평가가 거절
    assert writes


def test_delta_is_order_invariant_and_needs_three_runs():
    a, b = calibrate([0.7, 0.8, 0.9]), calibrate([0.9, 0.7, 0.8])
    assert a["delta"] == b["delta"] == pytest.approx(2 * 2 ** 0.5 * 0.1, abs=1e-4)
    assert calibrate([0.7, 0.8])["delta"] is None


def test_errors_are_not_abstentions_and_judge_key_uses_full_answer(tmp_path):
    from adapter import RagDomain
    d = RagDomain()
    unans = next(i for i in d.items.values() if not i["answerable"])
    assert d._reward(unans, {"abstained": True, "error": "RuntimeError()"})[0] == 0.0
    assert d._reward(unans, {"abstained": True, "reason": "insufficient_evidence"})[0] == 1.0
    assert d._reward(unans, {"abstained": True})[0] == 0.0       # 사유 없는 기권은 알 수 없음(R01)
    ans = next(i for i in d.items.values() if i["answerable"])
    head = "가" * 130
    calls = []
    d._judge_cache.clear()
    import adapter
    orig = adapter.judge_answer
    adapter.judge_answer = lambda client, item, answer: calls.append(answer) or (1.0, "ok")
    try:
        d._judge_correct(ans, head + "정답")
        d._judge_correct(ans, head + "오답")
    finally:
        adapter.judge_answer = orig
    assert len(calls) == 2                      # 앞 120자가 같아도 따로 채점


def test_fingerprint_changes_with_harness(tmp_path):
    from adapter import RagDomain
    d = RagDomain()
    import shutil
    shutil.copytree(ROOT / "domains" / "rag" / "harness", tmp_path / "domains" / "rag" / "harness",
                    ignore=shutil.ignore_patterns("__pycache__"))
    fp1 = d.fingerprint(tmp_path, ["a"], 2)
    cfg = tmp_path / "domains" / "rag" / "harness" / "config.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    assert d.fingerprint(tmp_path, ["a"], 2) != fp1


def test_thresholds_fail_without_baseline_or_metric(tmp_path):
    rep = tmp_path / "r.json"
    from tests.meta.test_eval_conditions import GOOD   # 문항별 결과까지 갖춘 합성 보고서(R02)
    ver = {k: GOOD[k] for k in ("goldenset_version", "evaluator", "judge", "ids", "n_unique",
                                "subset_sha256", "error_rate", "items")}
    rep.write_text(json.dumps(GOOD), encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "eval" / "check_thresholds.py"), str(rep), *a],  # noqa: E731
                                    capture_output=True, text=True, encoding="utf-8", errors="replace").returncode
    base = tmp_path / "base.json"
    assert run("--baseline", str(base)) == 1
    assert run("--baseline", str(base), "--init-baseline") == 0 and base.exists()
    assert run("--baseline", str(base)) == 0
    rep.write_text(json.dumps({"n": 12, "correct": 0.9, **ver}), encoding="utf-8")
    assert run("--baseline", str(base)) == 1                       # 필수 지표 누락
    rep.write_text(json.dumps({"n": 3, "correct": 0.9, "cite_precision": 1.0, **ver,
                               "abstain_answerable": 0.0, "abstain_unanswerable": 1.0}), encoding="utf-8")
    assert run("--baseline", str(base)) == 1                       # 표본 부족
