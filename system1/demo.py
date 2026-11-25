"""1장 데모: 게이트 4종을 순서대로 실행해 판정을 출력합니다.

    python3 -m system1.demo            # 로컬 대체 클라이언트
    DECISION_BASE_URL=http://localhost:3000 python3 -m system1.demo --remote
"""
from __future__ import annotations

import argparse
import os

from .gates import ROUTE_OPTIONS, escalate, faithfulness, route, sufficiency
from .local import HeuristicDecision
from .types import Citation, Passage


QUESTION = "환불 요청은 결제 후 며칠 이내에 해야 하나요?"
REFUND_TEXT = ("환불은 결제일로부터 14일 이내에 요청할 수 있다. "
               "개봉한 상품은 7일 이내에만 가능하다.")
PASSAGES = [Passage("acme:refund-policy#0", REFUND_TEXT, doc_id="acme:refund-policy")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true", help="DECISION_BASE_URL 사용")
    ap.add_argument("--question", default=QUESTION)
    a = ap.parse_args()

    if a.remote:
        from .client import DecisionClient
        base = os.environ.get("DECISION_BASE_URL")
        if not base:
            raise SystemExit("DECISION_BASE_URL이 필요합니다")
        client = DecisionClient(base, os.environ.get("DECISION_MODEL", "jev-1.13.0"))
    else:
        client = HeuristicDecision()

    r = route(client, a.question)
    print(f"G1 route      → {r.path:12s} reason={r.reason}")
    print(f"   criteria   → {', '.join(ROUTE_OPTIONS)}")

    s = sufficiency(client, a.question, PASSAGES)
    print(f"G2 sufficient → {s['verdict']:12s} p_yes={s['p_yes']:.3f} coverage={s['score']}")

    answer = "환불은 결제일로부터 14일 이내에 요청할 수 있습니다. [1]"
    f = faithfulness(client, answer, [Citation("acme:refund-policy#0", "결제일로부터 14일 이내", REFUND_TEXT)])
    print(f"G3 faithful   → ok={f['ok']} p_grounded={f['p_grounded']:.3f} "
          f"severity={f['severity']}")

    e = escalate(client, a.question, {"p_yes": s["p_yes"]},
                 {"p_grounded": f["p_grounded"], "severity": f["severity"]})
    print(f"G4 escalate   → {e['dest']:12s} p={e['p']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
