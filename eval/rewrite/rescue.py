#!/usr/bin/env python3
"""재작성 구제율 (6.3절).

G2가 rewrite 판정을 낸 케이스에서, 재검색이 정답 근거를 찾았는지 집계합니다.
재작성이 검색을 넓히기만 하고 정답을 못 찾으면 비용만 쓰는 장치입니다.

    python eval/rewrite/rescue.py --split evolve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "domains" / "rag"))

from harness.pipeline import RAGPipeline  # noqa: E402

DATA = ROOT / "domains" / "rag" / "data"


def hit(pipe: RAGPipeline, item: dict, question: str) -> bool:
    passages = pipe.retriever.search(question, item["tenant"], top_k=8)
    gold = {c["chunk_id"] for c in item["chunks"]}
    return bool({p.chunk_id for p in passages} & gold)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evolve")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    pipe = RAGPipeline.from_config(ROOT / "domains/rag/harness/config.yaml")
    items = {json.loads(l)["id"]: json.loads(l) for l in
             (DATA / "goldenset_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    split = json.loads((DATA / "split_rag.json").read_text(encoding="utf-8"))
    ids = split[a.split][: a.limit or None]

    flagged = rescued = degraded = 0
    for tid in ids:
        item = items[tid]
        if not item["chunks"]:
            continue
        res = pipe.answer_sync(item["question"], item["tenant"], "pro")
        if (res.get("decisions", {}).get("g2", {}) or {}).get("verdict") != "rewrite":
            continue
        flagged += 1
        before = hit(pipe, item, item["question"])
        after = hit(pipe, item, pipe._rewrite(item["question"]))
        if after and not before:
            rescued += 1
        if before and not after:
            degraded += 1

    out = {"split": a.split, "flagged": flagged, "rescued": rescued, "degraded": degraded,
           "rescue_rate": round(rescued / flagged, 3) if flagged else 0.0}
    print(json.dumps(out, ensure_ascii=False))
    path = ROOT / "eval" / "reports" / f"rewrite_{a.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
