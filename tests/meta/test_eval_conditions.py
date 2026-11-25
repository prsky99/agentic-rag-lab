"""평가 조건 회귀 시험(2차 검토 E03·E06).

* E03: 평가 지문이 하네스가 최종 적용한 실행 조건을 담는지, 조건이 바뀌면 이전 결과를
  다시 쓰지 않는지 확인합니다.
* E06: 임계값 검사가 현재 보고서와 기준선을 둘 다 검증하고, 버전이 다르면 비교하지 않는지
  확인합니다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "eval" / "check_thresholds.py"
UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}      # 윈도 콘솔 인코딩과 무관하게 읽습니다
IDS = [f"q_{i:04d}" for i in range(1, 13)]


def _sha(ids):
    import hashlib
    norm = json.dumps(sorted(set(ids)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _row(i: int) -> dict:
    """합성 문항별 결과: 1~9 정답, 10 오답(인용은 맞음), 11·12 답변 불가 정상 기권."""
    tid = IDS[i]
    if i >= 10:
        return {"id": tid, "answerable": False, "outcome": "abstained", "abstained": True,
                "reason": "insufficient_evidence", "error": None, "reward": 1.0,
                "correct": None, "judge": "skipped", "cite_precision": None}
    ok = i < 9
    return {"id": tid, "answerable": True, "outcome": "answered", "abstained": False,
            "reason": "", "error": None, "reward": 1.0 if ok else 0.35,
            "correct": 1.0 if ok else 0.0, "judge": "ok", "cite_precision": 1.0}


ITEMS = [_row(i) for i in range(12)]
GOOD = {"n": 12, "S": round((9 + 0.35 + 2) / 12, 4), "correct": 0.9, "cite_precision": 1.0,
        "abstain_answerable": 0.0, "abstain_unanswerable": 1.0, "error_rate": 0.0,
        "judge_errors": 0,
        "outcomes": {"answered": 10, "abstained": 2, "blocked": 0, "ops_failure": 0,
                     "budget_exceeded": 0, "unknown": 0, "error": 0},
        "goldenset_version": "goldenset_v1@sha256:aaaaaaaaaaaa",
        "evaluator": "rag-eval/5", "judge": "local/local-fallback/same-v1",
        "ids": IDS, "n_unique": 12, "subset_sha256": _sha(IDS), "items": ITEMS}


# ---------------------------------------------------------------------------
# E03 평가 지문: 최종 적용된 실행 조건
# ---------------------------------------------------------------------------
@pytest.fixture
def domain():
    from adapter import RagDomain
    return RagDomain()


_BASE: dict = {}                    # 환경 변수를 비운 상태의 지문(한 번만 계산합니다)


def _clear(monkeypatch):
    for name in ("DECISION_MODE", "DECISION_BASE_URL", "DECISION_MODEL", "RETRIEVER_URL",
                 "OPENAI_BASE_URL", "OPENAI_API_KEY", "GENERATOR_URL", "GENERATOR_API_KEY",
                 "GENERATOR_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_e03_fingerprint_holds_resolved_runtime(domain, monkeypatch):
    _clear(monkeypatch)
    rt = domain.fingerprint(ROOT, ["a"], 1)["runtime"]
    assert rt["decision"] == {"mode": "local", "model": "jev-1.13.0", "address": "",
                              "provider": "local_fallback"}
    assert rt["generation"]["mode"] == "extractive"
    assert rt["retrieval"]["backend"] == "local" and rt["retrieval"]["index_sha256"]
    assert rt["index_snapshot"]["name"] == "idx-2026-11-20"
    assert rt["config"]["budget"]["deadline_ms"] == 6000
    fp = domain.fingerprint(ROOT, ["a"], 1)
    assert fp["corpus_sha256"] and fp["evaluator"] == domain.EVALUATOR_VERSION


@pytest.mark.parametrize("env, path, want", [
    ({"DECISION_MODE": "offline"}, ("decision", "mode"), "offline"),
    ({"DECISION_BASE_URL": "http://decision:8000"}, ("decision", "mode"), "remote"),
    ({"DECISION_BASE_URL": "http://decision:8000"}, ("decision", "address"),
     "http://decision:8000"),
    # 우선순위: DECISION_MODE가 주소보다 먼저입니다(하네스 코드와 같은 해석).
    ({"DECISION_BASE_URL": "http://decision:8000", "DECISION_MODE": "local"},
     ("decision", "mode"), "local"),
    ({"RETRIEVER_URL": "http://retriever:8080"}, ("retrieval", "backend"), "http"),
    ({"RETRIEVER_URL": "http://retriever:8080"}, ("retrieval", "address"),
     "http://retriever:8080"),
    ({"GENERATOR_URL": "http://generator:8080"}, ("generation", "provider"), "generator"),
    ({"OPENAI_BASE_URL": "https://api.example/v1", "OPENAI_API_KEY": "k"},
     ("generation", "mode"), "remote"),
])
def test_e03_fingerprint_follows_effective_settings(domain, monkeypatch, env, path, want):
    _clear(monkeypatch)
    if "fp" not in _BASE:
        _BASE["fp"] = domain.fingerprint(ROOT, ["a"], 1)
    base = _BASE["fp"]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    fp = domain.fingerprint(ROOT, ["a"], 1)
    assert fp["runtime"][path[0]][path[1]] == want
    assert fp != base
    assert isinstance(fp["runtime"]["generation"]["has_api_key"], bool)   # 키 값은 싣지 않습니다


def test_e03_env_overridden_by_config_does_not_change_fingerprint(domain, monkeypatch):
    """config.yaml이 정한 모델 이름은 환경 변수보다 먼저입니다. 적용되지 않는 환경 변수는
    실행 조건을 바꾸지 않으므로 지문도 그대로입니다(환경 변수 이름이 아니라 최종 값을 봅니다)."""
    _clear(monkeypatch)
    base = domain.fingerprint(ROOT, ["a"], 1)
    monkeypatch.setenv("DECISION_MODEL", "jev-2.0.0")
    monkeypatch.setenv("GENERATOR_MODEL", "other-model")
    fp = domain.fingerprint(ROOT, ["a"], 1)
    assert fp["runtime"]["decision"]["model"] == "jev-1.13.0"
    assert fp == base


def test_e03_fingerprint_changes_with_corpus_and_config_value(domain, tmp_path, monkeypatch):
    import shutil
    _clear(monkeypatch)
    src = ROOT / "domains" / "rag"
    dst = tmp_path / "domains" / "rag"
    shutil.copytree(src / "harness", dst / "harness", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(src / "data" / "corpus", dst / "data" / "corpus")
    fp1 = domain.fingerprint(tmp_path, ["a"], 1)
    assert fp1["runtime"]["retrieval"]["chunks"] > 0
    doc = next((dst / "data" / "corpus").rglob("*.md"))
    doc.write_text(doc.read_text(encoding="utf-8") + "\n추가 문장입니다.\n", encoding="utf-8")
    fp2 = domain.fingerprint(tmp_path, ["a"], 1)
    assert fp2["runtime"]["retrieval"]["index_sha256"] != fp1["runtime"]["retrieval"]["index_sha256"]


def test_e03_mode_change_discards_previous_results(domain, tmp_path, monkeypatch):
    """검토자의 재현 조건: 같은 job 이름으로 local 다음에 offline을 실행합니다."""
    _clear(monkeypatch)
    ids = [next(i for i, it in domain.items.items() if it["answerable"])]
    out = tmp_path / "jobs" / "same" / "s0" / f"{ids[0]}.json"

    monkeypatch.setenv("DECISION_MODE", "local")
    domain.run(ROOT, tmp_path, "same", ids, 1)
    first = json.loads(out.read_text(encoding="utf-8"))
    assert first["status"] == "answered"

    monkeypatch.setenv("DECISION_MODE", "offline")
    domain.run(ROOT, tmp_path, "same", ids, 1)
    second = json.loads(out.read_text(encoding="utf-8"))
    assert second["status"] == "abstained" and second["reason"] == "g3_unavailable"

    # 조건이 같으면 완료된 결과를 그대로 씁니다(재개).
    stamp = out.stat().st_mtime_ns
    domain.run(ROOT, tmp_path, "same", ids, 1)
    assert out.stat().st_mtime_ns == stamp


# ---------------------------------------------------------------------------
# E06 임계값 검사: 두 보고서 검증, 버전 불일치 거절
# ---------------------------------------------------------------------------
def _check(tmp_path: Path, report, base, *args) -> subprocess.CompletedProcess:
    rep, bas = tmp_path / "report.json", tmp_path / "base.json"
    rep.write_text(json.dumps(report), encoding="utf-8")
    if base is not None:
        bas.write_text(json.dumps(base), encoding="utf-8")
    elif bas.exists():
        bas.unlink()
    return subprocess.run([sys.executable, str(CHECK), str(rep), "--baseline", str(bas), *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=UTF8)


def test_e06_valid_pair_passes(tmp_path):
    assert _check(tmp_path, GOOD, GOOD).returncode == 0


@pytest.mark.parametrize("change", [
    {"n": None},                        # n 누락
    {"n": "0"},                         # 문자열
    {"n": "12"},
    {"n": 12.0},                        # 정수가 아닌 수
    {"n": True},
    {"n": 0},                           # 최솟값 미만
    {"n": 9},
    {"correct": "0.9"},                 # 수가 아닌 지표
    {"correct": 1.2},                   # 범위 밖
    {"cite_precision": -0.1},
    {"abstain_unanswerable": None},
    {"abstain_answerable": float("nan")},
    {"goldenset_version": None},
    {"goldenset_version": ""},
    {"evaluator": None},
    {"error_rate": None},               # 실행 오류율 누락(E05)
    {"error_rate": 10 ** 400},          # float로 바꿀 수 없는 큰 정수(RT11)
    {"subset_sha256": None},            # 문항 목록 해시 누락(E06)
    {"ids": None},
    {"ids": IDS[:11]},                  # n과 개수가 다름
    {"ids": ["q_0001"] * 11 + ["q_0012"]},   # 중복 문항
])
@pytest.mark.parametrize("side", ["report", "baseline"])
def test_e06_invalid_report_or_baseline_is_rejected(tmp_path, change, side):
    bad = {k: v for k, v in {**GOOD, **change}.items() if v is not None}
    report, base = (bad, GOOD) if side == "report" else (GOOD, bad)
    proc = _check(tmp_path, report, base)
    assert proc.returncode == 1, proc.stdout
    assert ("현재 보고서" if side == "report" else "기준선") in proc.stdout


@pytest.mark.parametrize("key, other", [
    ("goldenset_version", "goldenset_v2@sha256:bbbbbbbbbbbb"),
    ("evaluator", "rag-eval/4"),
    ("judge", "remote/jev-1.13.0/same-v1"),      # 심판이 다르면 비교하지 않음(R03)
])
def test_e06_version_mismatch_is_not_comparable(tmp_path, key, other):
    # 기준선 쪽 지표가 더 나빠도(=현재가 더 좋아 보여도) 비교하지 않습니다.
    proc = _check(tmp_path, GOOD, {**GOOD, key: other})
    assert proc.returncode == 1
    assert "비교할 수 없음" in proc.stdout and "d +" not in proc.stdout


def test_e06_init_baseline_requires_valid_report(tmp_path):
    proc = _check(tmp_path, {**GOOD, "n": "0"}, None, "--init-baseline")
    assert proc.returncode == 1 and not (tmp_path / "base.json").exists()
    proc = _check(tmp_path, {k: v for k, v in GOOD.items() if k != "goldenset_version"},
                  None, "--init-baseline")
    assert proc.returncode == 1 and not (tmp_path / "base.json").exists()
    assert _check(tmp_path, GOOD, None, "--init-baseline").returncode == 0
    assert (tmp_path / "base.json").exists()


def test_e06_run_subset_report_and_committed_baseline_are_comparable(tmp_path):
    out = tmp_path / "report.json"
    proc = subprocess.run([sys.executable, str(ROOT / "eval" / "run_subset.py"),
                           "--subset", str(ROOT / "eval" / "ci" / "regression_subset.json"),
                           "--config", str(ROOT / "domains/rag/harness/config.yaml"),
                           "--out", str(out)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", cwd=str(ROOT), env=UTF8)
    assert proc.returncode == 0, proc.stderr[-500:]
    report = json.loads(out.read_text(encoding="utf-8"))
    from adapter import RagDomain, goldenset_version
    assert report["goldenset_version"] == goldenset_version()
    assert report["evaluator"] == RagDomain.EVALUATOR_VERSION
    assert isinstance(report["n"], int) and report["usage_unknown_calls"] == 0
    proc = subprocess.run([sys.executable, str(CHECK), str(out), "--baseline",
                           str(ROOT / "eval" / "reports" / "base_ragas.json")],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=UTF8)
    assert proc.returncode == 0, proc.stdout
