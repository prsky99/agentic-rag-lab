"""S_hat / C_hat 집계 (7.3절).

    S_hat(H) = 1/(k·|D|) · Σ_x Σ_j r(x, τ_x^(j))
    C_hat(H) = 1/(k·|D|) · Σ_x Σ_j c(τ_x^(j))

보상은 도메인이 계산합니다. 코어는 보상의 의미와 무관하게 평균만 냅니다.

산출물이 없는 트라이얼(누락)은 S에서는 0점으로 세고, C의 분모에서는 뺍니다. 그래서 누락 수와
비용을 잰 트라이얼 수를 따로 남깁니다(G02). 모두 누락되면 S=0·C=0이 되는데, 이 C=0은 실제
0토큰이 아니라 비용 미측정이므로 ``summary_problems``가 기준선 등록을 막습니다.
"""
from __future__ import annotations

import math
from pathlib import Path

from ._log import log, read_json, write_json
from .domain import Domain, EvalSummary
from .gitops import Workspace


def evaluate(domain: Domain, root: Path, runs_dir: Path, job: str,
             ids: list[str], k: int, announce: bool = True) -> EvalSummary:
    root, runs_dir = Path(root), Path(runs_dir)
    domain.run(root, runs_dir, job, ids, k)
    per_task, extra = domain.score(runs_dir, job, ids, k)

    rewards = [r for tr in per_task.values() for r in tr.rewards]
    tokens = [t for tr in per_task.values() for t in tr.tokens if t is not None]
    missing = sum(tr.missing for tr in per_task.values())
    total = sum(len(tr.rewards) for tr in per_task.values())
    expected = len(ids) * int(k)

    summary = EvalSummary(
        S=sum(rewards) / len(rewards) if rewards else 0.0,
        C=sum(tokens) / len(tokens) if tokens else 0.0,
        k=k,
        per_task=per_task,
        extra=extra,
        n=len(per_task),
        # 분모가 0이면(산출물·채점 결과가 전혀 없으면) 누락 100%로 둡니다. 0%로 두면 정상처럼 보입니다.
        missing_frac=(missing / total) if total else 1.0,
        trials_expected=expected,
        trials_done=total - missing,
        cost_trials=len(tokens),
        # 비교 조건과 하네스 해시를 함께 남깁니다. 기준선 비교·δ 보정이 이 값을 대조합니다(F01·F02).
        conditions=domain.conditions(root, ids, k),
        harness_tree=Workspace(root, runs_dir, f"domains/{domain.name}/{domain.harness_path}").tree_hash(),
    )
    job_dir = runs_dir / "jobs" / job
    write_json(job_dir / "eval.json", {
        "job": job, "S": round(summary.S, 6), "C": round(summary.C, 2),
        "k": k, "n": summary.n, "missing_frac": round(summary.missing_frac, 4),
        "trials_expected": summary.trials_expected, "trials_done": summary.trials_done,
        "cost_trials": summary.cost_trials,
        "extra": summary.extra,
        "conditions": summary.conditions, "harness_tree": summary.harness_tree,
    })
    if announce:
        log(domain.name, f"{job}: S={summary.S:.4f} C={summary.C:.1f} "
                         f"missing={missing}/{total}")
    return summary


def _int(val) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _real(val) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and         (abs(val) <= 2 ** 53 if isinstance(val, int) else math.isfinite(val))


def summary_problems(summary: EvalSummary, ids: list[str] | None = None,
                     k: int | None = None) -> list[str]:
    """기준선으로 쓸 수 있는 집계인지(G02·G04). 사유 목록이 비어 있어야 씁니다.

    기준선은 누락 0이 원칙입니다. 예상 문항 수·k와 맞고, 예상 트라이얼(문항×k)이 모두 산출물을
    내고 모두 비용을 쟀어야 합니다. S는 0~1, C는 0 이상의 유한한 수여야 합니다. C=0은 모든
    트라이얼의 비용을 쟀을 때만 "실제 0토큰"으로 인정합니다. 도메인 검사(심판 장애 등)는 따로 봅니다.
    """
    v: list[str] = []
    if not _real(summary.S) or not 0.0 <= summary.S <= 1.0:
        v.append(f"S={summary.S!r}가 0~1의 유한한 수가 아닙니다")
    if not _real(summary.C) or summary.C < 0:
        v.append(f"C={summary.C!r}가 0 이상의 유한한 수가 아닙니다")
    if not _int(summary.k) or summary.k < 1:
        v.append(f"k={summary.k!r}가 양의 정수가 아닙니다")
    if not _int(summary.n) or summary.n < 1:
        v.append(f"n={summary.n!r}가 양의 정수가 아닙니다")
    if k is not None and summary.k != int(k):
        v.append(f"k {summary.k!r} != 설정 {k}")
    if ids is not None and summary.n != len(ids):
        v.append(f"n {summary.n!r} != 예상 문항 {len(ids)}")
    if not _real(summary.missing_frac) or summary.missing_frac != 0:
        v.append(f"누락 트라이얼 missing_frac={summary.missing_frac!r}(기준선은 0이어야 합니다)")
    exp, done, cost = summary.trials_expected, summary.trials_done, summary.cost_trials
    if not all(_int(x) for x in (exp, done, cost)) or exp < 1:
        v.append("트라이얼 수 기록 없음(이전 판 결과) → baseline 다시 측정")
    else:
        if _int(summary.k) and _int(summary.n) and exp != summary.n * summary.k:
            v.append(f"예상 트라이얼 {exp} != 문항 {summary.n}×k {summary.k}")
        if done != exp:
            v.append(f"산출물 {done}/{exp} 트라이얼(누락 {exp - done})")
        if cost != exp:
            v.append(f"비용 측정 {cost}/{exp} 트라이얼(C 미측정분 있음)")
    return v


def load_summary(runs_dir: Path, job: str) -> dict:
    return read_json(Path(runs_dir) / "jobs" / job / "eval.json", {}) or {}


def p95_latency(runs_dir: Path, job: str) -> float:
    """job 산출물에서 p95 지연을 읽습니다(가드용)."""
    vals: list[float] = []
    for path in sorted((Path(runs_dir) / "jobs").glob(f"{job}/s[0-9]*/*.json")):
        try:
            data = read_json(path, {}) or {}
        except Exception:                      # 깨진 산출물은 지표에서 제외
            continue
        if not isinstance(data, dict):
            continue
        ms = data.get("latency_ms")
        if isinstance(ms, (int, float)):
            vals.append(float(ms))
    if not vals:
        return 0.0
    vals.sort()
    idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
    return vals[idx]
