#!/usr/bin/env python3
"""게이트만 따로 채점하기 (4.7절).

    python -m eval.gates.run_gates                    # 로컬 대체 판정(system1.local)
    DECISION_BASE_URL=http://localhost:8000 python -m eval.gates.run_gates --remote

동봉 골든셋(42문항)에서 G2·G3 입력을 만듭니다. 답변 가능 문항은 자기 정답 문단을
'충분' 양성으로, 다른 문항의 문단을 붙인 짝을 '불충분' 음성으로 씁니다.
결과와 임계값 스윕은 ``eval/reports/gates.json``에 남습니다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.gates.thresholds import GateCase, ece, sweep  # noqa: E402

DATA = ROOT / "domains" / "rag" / "data"


@dataclass
class GoldenItem:
    qid: str
    question: str
    passages: list[dict]
    answer_gold: str | None
    label_sufficient: bool
    label_grounded: bool | None
    citations: list[dict] | None


def evaluate_gates(client, items: list[GoldenItem], policy) -> dict:
    from system1.gates import answerability, faithfulness
    from system1.types import Citation, Passage
    rows = []
    for it in items:
        passages = [Passage(**p) for p in it.passages]
        g2 = answerability(client, it.question, passages, policy.answer_floor)
        row = {
            "qid": it.qid,
            "g2_p": g2["p_yes"],
            "g2_verdict": g2["verdict"],
            "g2_label": it.label_sufficient,
        }
        if it.answer_gold and it.citations:
            g3 = faithfulness(client, it.answer_gold,
                              [Citation(**c) for c in it.citations], policy.faith_floor)
            row.update({"g3_p": g3["p_grounded"], "g3_ok": g3["ok"],
                        "g3_label": it.label_grounded})
        rows.append(row)
    return {"n": len(rows), "rows": rows}


def load_items(path: Path) -> list[GoldenItem]:
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
           if line.strip()]
    answerable = [r for r in raw if r.get("answerable") and r.get("chunks")]
    items: list[GoldenItem] = []
    for i, r in enumerate(raw):
        chunks = [{"chunk_id": c["chunk_id"], "text": c["text"], "doc_id": c["doc_id"]}
                  for c in r.get("chunks") or []]
        ok = bool(r.get("answerable") and chunks)
        cites = ([{"chunk_id": c["chunk_id"], "span": r["reference"], "text": c["text"]}
                  for c in chunks] if ok else None)
        items.append(GoldenItem(r["id"], r["question"], chunks,
                                r.get("reference") if ok else None, ok,
                                True if ok else None, cites))
        if ok and len(answerable) > 1:       # 음성 짝: 다른 문항의 문단을 붙입니다
            other = answerable[(answerable.index(r) + 1) % len(answerable)]
            if {c["doc_id"] for c in other["chunks"]} != {c["doc_id"] for c in chunks}:
                neg = [{"chunk_id": c["chunk_id"], "text": c["text"], "doc_id": c["doc_id"]}
                       for c in other["chunks"]]
                items.append(GoldenItem(f"{r['id']}-neg", r["question"], neg,
                                        None, False, None, None))
    return items


def main() -> int:
    from system1.policy import GatePolicy
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true", help="DECISION_BASE_URL 사용")
    ap.add_argument("--goldenset", default=str(DATA / "goldenset_v1.jsonl"))
    a = ap.parse_args()

    if a.remote:
        if not os.environ.get("DECISION_BASE_URL"):
            raise SystemExit("--remote에는 DECISION_BASE_URL이 필요합니다 "
                             "(예: DECISION_BASE_URL=http://localhost:8000)")
        from system1.client import DecisionClient
        client = DecisionClient(os.environ["DECISION_BASE_URL"],
                                os.environ.get("DECISION_MODEL", "jev-1.13.0"))
    else:
        from system1.local import HeuristicDecision
        client = HeuristicDecision()

    policy = GatePolicy()
    result = evaluate_gates(client, load_items(Path(a.goldenset)), policy)
    cases = [GateCase(r["qid"], r["g2_p"], r["g2_label"]) for r in result["rows"]]
    rows, best = sweep(cases)
    result.update({"policy": asdict(policy), "g2_best": best, "g2_ece": round(ece(cases), 4),
                   "g2_sweep": [r for r in rows if round(r["t"] * 100) % 5 == 0]})
    print(f"G2 n={len(cases)}  best t={best['t']:.2f} cost={best['cost']:.1f} "
          f"precision={best['precision']:.3f} recall={best['recall']:.3f} "
          f"ECE={result['g2_ece']:.3f}")
    out = ROOT / "eval" / "reports" / "gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
