#!/usr/bin/env python3
"""비용 규칙의 β1 유도 (8.8절).

"ΔS 0.01이 회사에 얼마짜리인가"를 월 비용에서 역산합니다. β를 감으로 정하면
비용 규칙은 취향이 되고, 역산하면 협상 가능한 숫자가 됩니다.

    python3 tools/derive_beta.py --queries 1000000 --tokens 4000 --won-per-mtok 3000
"""
from __future__ import annotations

import argparse


def derive_beta(monthly_queries: float, tokens_per_query: float, won_per_mtok: float,
                point_value_frac: float, beta0: float) -> dict:
    monthly_cost = monthly_queries * tokens_per_query / 1e6 * won_per_mtok
    value_per_point = monthly_cost * point_value_frac      # ΔS=0.01당 원
    dC_per_point = value_per_point / monthly_cost          # = point_value_frac
    beta1 = dC_per_point / 0.01
    return {"monthly_cost_krw": round(monthly_cost),
            "beta0": beta0, "beta1": round(beta1, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=float, default=1_000_000)
    ap.add_argument("--tokens", type=float, default=4000)
    ap.add_argument("--won-per-mtok", type=float, default=3000)
    ap.add_argument("--point-value-frac", type=float, default=0.10)
    ap.add_argument("--beta0", type=float, default=0.03)
    a = ap.parse_args()
    print(derive_beta(a.queries, a.tokens, a.won_per_mtok, a.point_value_frac, a.beta0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
