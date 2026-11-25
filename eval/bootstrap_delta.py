#!/usr/bin/env python3
"""δ를 부트스트랩으로 측정하기 (9.6절).

rrsi/calibrate.py(8.9절)와 같은 척도의 δ를 계산합니다.

* calibrate.py: 같은 하네스를 여러 번 실행한 점수 S_1..S_n의 **쌍별 차이** 표준편차
  sd(S_i - S_j)를 씁니다. 두 실행이 독립이면 sd(S_i - S_j) = √2 · σ (σ: 한 번 측정한 S의 표준편차).
* 여기: 한 실행 안에서 과제를 복원추출해 σ를 추정합니다. 그래서 차이의 척도로 옮기려면
  √2를 곱합니다. δ = z · √2 · σ_rms.

과제 리샘플링은 문항 표집에 따른 변동까지 포함하므로, 보통 calibrate.py보다 δ가
크게(보수적으로) 나옵니다. baseline을 여러 번 실행하기 어려울 때의 대안으로 씁니다.

입력은 rrsi.domain.EvalSummary(``per_task``: 과제 → TaskResult(rewards), ``k``)입니다.
"""
from __future__ import annotations

import math
import random


def bootstrap_S(per_task: dict, k: int, n: int = 2000, seed: int = 0) -> dict:
    """과제를 복원추출해 S_hat의 표본분포를 만듭니다."""
    rng = random.Random(seed)
    tasks = list(per_task.values())
    vals = []
    for _ in range(n):
        num = den = 0.0
        for _ in range(len(tasks)):
            tr = rng.choice(tasks)
            rs = tr.rewards[:k]
            num += sum(rs)
            den += len(rs)
        vals.append(num / den if den else 0.0)
    vals.sort()
    return {"mean": sum(vals) / len(vals),
            "sd": (vals[int(0.84 * n)] - vals[int(0.16 * n)]) / 2,   # 16~84 백분위 폭의 절반 ≈ 1σ
            "p05": vals[int(0.05 * n)], "p95": vals[int(0.95 * n)]}


def delta_from_runs(evals: list, z: float = 2.0) -> dict:
    """실행마다 σ를 구해 제곱평균(RMS)으로 합치고, 차이 척도(√2)로 옮겨 δ를 계산합니다."""
    sds = [bootstrap_S(e.per_task, e.k)["sd"] for e in evals]
    sigma = math.sqrt(sum(s * s for s in sds) / len(sds))      # 한 번 측정한 S의 표준편차
    sd_null = math.sqrt(2) * sigma                             # 두 실행 차이의 표준편차
    return {"delta": round(z * sd_null, 5), "sd_null": round(sd_null, 5),
            "sigma": round(sigma, 5), "z": z, "method": "bootstrap", "n": len(evals)}


def main() -> int:
    """baseline job 산출물(runs/<도메인>/jobs/<job>/)로 부트스트랩 δ를 계산합니다.

        python eval/bootstrap_delta.py --jobs base
        python eval/bootstrap_delta.py --jobs base1,base2,base3
    """
    import argparse
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    ap = argparse.ArgumentParser(description="부트스트랩 δ (9.6절)")
    ap.add_argument("--domain", default="rag")
    ap.add_argument("--jobs", default="base", help="쉼표로 구분한 baseline job 이름")
    ap.add_argument("--z", type=float, default=2.0)
    a = ap.parse_args()

    from rrsi.cli import load_domain
    domain = load_domain(root, a.domain)
    runs_dir = root / "runs" / domain.name
    evals = []
    for job in [j.strip() for j in a.jobs.split(",") if j.strip()]:
        job_dir = runs_dir / "jobs" / job
        if not (job_dir / "eval.json").exists():
            print(f"[bootstrap] {job_dir}/eval.json 없음. 먼저 "
                  f"`python rrsi.py --domain {a.domain} baseline --job {job}`")
            return 1
        k = int(json.loads((job_dir / "eval.json").read_text(encoding="utf-8"))["k"])
        ids = [t["id"] for t in json.loads((job_dir / "_ids.json").read_text(encoding="utf-8"))]
        per_task, _ = domain.score(runs_dir, job, ids, k)
        evals.append(SimpleNamespace(per_task=per_task, k=k))
    print(json.dumps(delta_from_runs(evals, z=a.z), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
