#!/usr/bin/env python3
"""질문 유형별 재현율 (3.1절).

dense / bm25 / hybrid 세 실행을 같은 문항으로 실행해 유형별 recall@k를 비교합니다.
"평균이 올랐다"가 아니라 "어느 유형이 올랐는가"를 확인합니다.

    python eval/recall_by_type.py --split evolve --k 50
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "domains" / "rag"))

import yaml  # noqa: E402
from harness.retrieve import build_retriever, hash_vector, cosine, tokenize  # noqa: E402

DATA = ROOT / "domains" / "rag" / "data"


@dataclass
class Case:
    qid: str
    question: str
    qtype: str          # lexical | semantic | mixed
    gold_doc_ids: set[str]


def _doc(cid: str) -> str:
    """청크 ID(`acme/refund-policy#1`) → 문서 ID. 정답이 문서 단위라 비교 전에 맞춥니다."""
    return cid.split("#", 1)[0]


def recall_at_k(retrieved: list[str], gold: set[str]) -> float:
    retrieved = [_doc(r) for r in retrieved]
    if not gold:
        return 0.0
    return len(set(retrieved) & gold) / len(gold)


def _bm25_only(index, question: str, tenant: str, k: int) -> list[str]:
    passages = index.search(question, tenant, top_k=k * 2)
    q = set(tokenize(question))
    ranked = sorted(passages, key=lambda p: -len(q & set(tokenize(p.text))))
    return [p.chunk_id for p in ranked[:k]]


def _dense_only(index, question: str, tenant: str, k: int) -> list[str]:
    allowed = index._tenants(tenant)
    qv = hash_vector(question)
    scored = [(cosine(qv, index._vecs[i]), index.passages[i].chunk_id)
              for i, p in enumerate(index.passages) if p.tenant in allowed]
    scored.sort(key=lambda x: -x[0])
    return [cid for _, cid in scored[:k]]


def report(cases: list[Case], run) -> dict[str, dict[str, float]]:
    agg: dict[str, list[float]] = defaultdict(list)
    for c in cases:
        agg[c.qtype].append(recall_at_k(run(c.question), c.gold_doc_ids))
    out = {}
    for qtype, vals in agg.items():
        out[qtype] = {"n": len(vals), "recall@50": round(sum(vals) / len(vals), 3)}
    allv = [v for vals in agg.values() for v in vals]
    out["all"] = {"n": len(allv), "recall@50": round(sum(allv) / len(allv), 3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evolve")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--config", default=str(ROOT / "domains/rag/harness/config.yaml"))
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
    ret_cfg = dict(cfg.get("retrieval") or {})
    ret_cfg["corpus_dir"] = str(ROOT / "domains/rag/data/corpus")
    index = build_retriever(ret_cfg, ROOT / "domains/rag/harness")

    split = json.loads((DATA / "split_rag.json").read_text(encoding="utf-8"))
    items = {json.loads(l)["id"]: json.loads(l)
             for l in (DATA / "goldenset_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    cases = []
    for tid in split.get(a.split, []):
        item = items[tid]
        gold = {c["doc_id"] for c in item.get("chunks") or []}
        if not gold:
            continue
        qtype = "lexical" if item["type"] == "single_fact" else "mixed"
        if item["type"] == "multi_condition":
            qtype = "semantic"
        cases.append(Case(tid, item["question"], qtype, gold))

    runs = {
        "dense": lambda q: _dense_only(index, q, "acme", a.k),
        "bm25": lambda q: _bm25_only(index, q, "acme", a.k),
        "hybrid": lambda q: [p.chunk_id for p in index.search(q, "acme", top_k=a.k)],
    }
    rows = {name: report(cases, fn) for name, fn in runs.items()}
    print(f"{'qtype':<10}{'n':>5}{'dense':>8}{'bm25':>8}{'hybrid':>8}")
    for qtype in list(rows["hybrid"].keys()):
        n = rows["hybrid"][qtype]["n"]
        print(f"{qtype:<10}{n:>5}"
              f"{rows['dense'][qtype]['recall@50']:>8.3f}"
              f"{rows['bm25'][qtype]['recall@50']:>8.3f}"
              f"{rows['hybrid'][qtype]['recall@50']:>8.3f}")
    out = ROOT / "eval" / "reports" / f"recall_by_type_{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
