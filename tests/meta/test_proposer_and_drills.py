"""제안자 입력(N-E02)과 장애 훈련 스크립트(D09) 회귀 시험."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rrsi.history import History
from rrsi.propose import DOC_LIMIT, EditError, LLMProposer, read_domain_doc

ROOT = Path(__file__).resolve().parents[2]


def _root(tmp_path: Path) -> Path:
    dom = tmp_path / "domains" / "rag"
    (dom / "harness").mkdir(parents=True)
    (dom / "data").mkdir()
    (dom / "harness" / "config.yaml").write_text("top_k: 8\n", encoding="utf-8")
    (dom / "SKILL.md").write_text("# 규칙\n- 금지: 정답 문자열 넣기\n", encoding="utf-8")
    (dom / "PATTERNS.md").write_text("# 패턴\n" + "가" * (DOC_LIMIT + 500), encoding="utf-8")
    (dom / "data" / "goldenset_v1.jsonl").write_text('{"reference": "비밀 정답"}\n', encoding="utf-8")
    return tmp_path


def test_llm_proposer_sends_rule_pattern_bodies_and_edit_originals(tmp_path):
    root = _root(tmp_path)
    hist = History(tmp_path / "history.jsonl")
    hist.append({"t": 0, "label": "r0A", "components": ["gate_policy"], "accepted": False,
                 "reason": "guard: over-abstention",
                 "edits": ["domains/rag/harness/config.yaml::top_k: 8->top_k: 4"]})
    prop = LLMProposer([], hist, base_url="http://x", api_key="k")
    seen = []
    prop._ask = lambda context: seen.append(json.loads(context)) or None   # 모델 대신 입력만 받습니다
    prop.propose(root, "r1A", 1, 1)
    ctx = seen[0]
    assert "금지: 정답 문자열 넣기" in ctx["skill"]                         # 경로 안내가 아니라 본문
    assert ctx["patterns"].startswith("# 패턴") and ctx["patterns"].endswith("(이하 생략)")
    assert len(ctx["patterns"]) <= DOC_LIMIT + 20                          # 크기 상한
    assert ctx["recent_edits"][0]["edits"] == ["domains/rag/harness/config.yaml::top_k: 8->top_k: 4"]
    assert "over-abstention" in ctx["history"] and ctx["config"] == "top_k: 8\n"
    assert "비밀 정답" not in json.dumps(ctx, ensure_ascii=False)          # 평가 데이터는 넣지 않음


def test_proposer_document_reader_is_limited_to_named_domain_docs(tmp_path):
    root = _root(tmp_path)
    for name in ("data/goldenset_v1.jsonl", "../../secret.txt", "harness/config.yaml"):
        with pytest.raises(EditError):
            read_domain_doc(root, name)
    assert read_domain_doc(tmp_path / "missing", "SKILL.md") == ""


def test_recent_edits_are_truncated(tmp_path):
    hist = History(tmp_path / "h.jsonl")
    hist.append({"t": 0, "edits": ["x" * 1000]})
    assert len(hist.recent_edits(limit=400)[0]["edits"][0]) < 420


# ---------------------------------------------------------------------------
# D09 장애 훈련: 복구 실패는 전체 실패, k6는 본문까지 검사하고 checks 임계값으로 실패
# ---------------------------------------------------------------------------
def _bash():
    path = shutil.which("bash")
    if not path or (os.name == "nt" and "system32" in path.lower()):   # WSL 런처는 쓰지 않습니다
        pytest.skip("bash 없음")
    return path


@pytest.mark.parametrize("fail_on, prior, want", [
    ("rollout", 0, 1),      # 앞 검사는 모두 통과했지만 복구 실패 → 실패
    ("scale", 0, 1),
    ("", 0, 0),             # 복구 성공 → 앞 결과 그대로
    ("", 3, 3),             # 앞 검사 실패 코드는 그대로 유지
])
def test_verify_restore_failure_fails_whole_run(tmp_path, fail_on, prior, want):
    bash = _bash()
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "kubectl").write_text(
        "#!/usr/bin/env bash\n"
        f'case " $* " in *" {fail_on or "__none__"} "*) exit 1;; esac\nexit 0\n',
        encoding="utf-8", newline="\n")
    (fake / "kubectl").chmod(0o755)
    src = (ROOT / "tests" / "verify.sh").read_text(encoding="utf-8")
    func = src[src.index("restore_decision() {"):]
    func = func[:func.index("\n}\n") + 3]
    script = tmp_path / "t.sh"
    script.write_text("set -euo pipefail\nPREV=1\n" + func +
                      f"trap restore_decision EXIT\n( exit {prior} ) || exit {prior}\n",
                      encoding="utf-8", newline="\n")
    env = {**os.environ, "PATH": f"{fake.as_posix()}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.run([bash, "-c", f'export PATH="{_posix(fake)}:$PATH"; bash "{_posix(script)}"'],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    assert proc.returncode == want, proc.stdout + proc.stderr
    assert ("[FAIL] decision 복구 실패" in proc.stdout) == bool(fail_on)


def _posix(p: Path) -> str:
    s = p.as_posix()
    if os.name == "nt" and len(s) > 1 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


def test_decision_down_k6_checks_body_and_fails_on_check_rate():
    js = (ROOT / "load" / "decision-down.js").read_text(encoding="utf-8")
    assert "thresholds: { checks: ['rate==1.0'] }" in js
    assert "b.abstained === true" in js and "b.reason === REASON" in js
    assert "g3_unavailable" in js and "decision_unavailable" in js


def test_recent_edits_group_candidates_by_round(tmp_path):
    """한 라운드에 후보가 여럿이어도 '최근 8라운드'는 라운드 8개입니다(R05)."""
    hist = History(tmp_path / "h.jsonl")
    for t in range(10):                       # 라운드 10개 × 후보 2개 = 기록 20개
        for c in "AB":
            hist.append({"t": t, "label": f"r{t}{c}", "edits": [f"f::{t}{c}"], "accepted": False})
    edits = hist.recent_edits()
    assert [e["t"] for e in edits] == [t for t in range(2, 10) for _ in "AB"]
    assert len({e["t"] for e in edits}) == 8 and len(edits) == 16
    assert hist.to_prompt().count("\n") + 1 == 16 and "t=1 " not in hist.to_prompt()
    assert {e["t"] for e in hist.recent_edits(rounds=1)} == {9}
