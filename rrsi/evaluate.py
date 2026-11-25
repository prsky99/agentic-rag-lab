"""S_hat / C_hat 집계 (7.3절).

    S_hat(H) = 1/(k·|D|) · Σ_x Σ_j r(x, τ_x^(j))
    C_hat(H) = 1/(k·|D|) · Σ_x Σ_j c(τ_x^(j))

보상은 도메인이 계산합니다. 코어는 보상의 의미와 무관하게 평균만 냅니다.
"""
from __future__ import annotations

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

    summary = EvalSummary(
        S=sum(rewards) / len(rewards) if rewards else 0.0,
        C=sum(tokens) / len(tokens) if tokens else 0.0,
        k=k,
        per_task=per_task,
        extra=extra,
        n=len(per_task),
        missing_frac=(missing / total) if total else 0.0,
        # 비교 조건과 하네스 해시를 함께 남깁니다. 기준선 비교·δ 보정이 이 값을 대조합니다(F01·F02).
        conditions=domain.conditions(root, ids, k),
        harness_tree=Workspace(root, runs_dir, f"domains/{domain.name}/{domain.harness_path}").tree_hash(),
    )
    job_dir = runs_dir / "jobs" / job
    write_json(job_dir / "eval.json", {
        "job": job, "S": round(summary.S, 6), "C": round(summary.C, 2),
        "k": k, "n": summary.n, "missing_frac": round(summary.missing_frac, 4),
        "extra": summary.extra,
        "conditions": summary.conditions, "harness_tree": summary.harness_tree,
    })
    if announce:
        log(domain.name, f"{job}: S={summary.S:.4f} C={summary.C:.1f} "
                         f"missing={missing}/{total}")
    return summary


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
