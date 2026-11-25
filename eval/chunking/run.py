#!/usr/bin/env python3
"""청킹 변형 비교 (3.3절).

같은 코퍼스를 네 가지 변형으로 잘라 recall@8과 색인 비용(청크 수·평균 토큰)을
비교합니다. "더 잘게 쪼개면 좋다"는 가정을 숫자로 반박하거나 지지하기 위한 표입니다.

    python eval/chunking/run.py
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "domains" / "rag"))

from harness.retrieve import LocalIndex, Passage, load_corpus  # noqa: E402

DATA = ROOT / "domains" / "rag" / "data"


@dataclass
class Variant:
    name: str
    max_tokens: int
    overlap: int
    strategy: str


VARIANTS = [
    Variant("fixed-400", 400, 40, "fixed"),
    Variant("fixed-800", 800, 80, "fixed"),
    Variant("struct-700", 700, 80, "structured"),
    Variant("struct-1000", 1000, 100, "structured"),
]


def tokens(text: str) -> int:
    return max(1, len(text) // 2)


def split_fixed(text: str, max_tokens: int, overlap: int) -> list[str]:
    words = text.split()
    out, cur, cur_tok = [], [], 0
    for w in words:
        cur.append(w)
        cur_tok += max(1, len(w) // 2)
        if cur_tok >= max_tokens:
            out.append(" ".join(cur))
            keep = max(0, len(cur) * overlap // max_tokens)
            cur = cur[-keep:] if keep else []
            cur_tok = tokens(" ".join(cur))
    if cur:
        out.append(" ".join(cur))
    return out


def split_structured(text: str, max_tokens: int, overlap: int) -> list[str]:
    """제목 단위로 나누고, 상한을 넘으면 문단 단위로만 자릅니다."""
    blocks = re.split(r"\n(?=#{2,3}\s)", "\n" + text)
    out = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if tokens(b) <= max_tokens:
            out.append(b)
            continue
        paras = [p for p in b.split("\n\n") if p.strip()]
        cur: list[str] = []
        for p in paras:
            if cur and tokens("\n\n".join(cur + [p])) > max_tokens:
                out.append("\n\n".join(cur))
                cur = [p]
            else:
                cur.append(p)
        if cur:
            out.append("\n\n".join(cur))
    return out


def build_index(variant: Variant) -> LocalIndex:
    passages = []
    for doc in load_corpus(DATA / "corpus"):
        body = doc["body"]
        parts = (split_fixed(body, variant.max_tokens, variant.overlap)
                 if variant.strategy == "fixed"
                 else split_structured(body, variant.max_tokens, variant.overlap))
        for i, text in enumerate(parts):
            passages.append(Passage(chunk_id=f"{doc['doc_id']}#{i}", doc_id=doc["doc_id"],
                                    tenant=doc["tenant"], title=doc["title"],
                                    section="", text=text))
    return LocalIndex(passages, {"tenant_aliases": {"demo": ["acme"], "other": ["globex"]}})


def evaluate(variant: Variant, cases: list[dict]) -> dict:
    index = build_index(variant)
    hit = 0
    for c in cases:
        gold_docs = {ch["doc_id"] for ch in c["chunks"]}
        got = {p.doc_id for p in index.search(c["question"], c["tenant"], top_k=8)}
        hit += bool(got & gold_docs)
    texts = [p.text for p in index.passages]
    return {
        "variant": variant.name,
        "recall@8": round(hit / len(cases), 3) if cases else 0.0,
        "chunks": len(texts),
        "avg_tokens": round(sum(tokens(t) for t in texts) / max(1, len(texts)), 1),
    }


def main() -> int:
    items = [json.loads(l) for l in
             (DATA / "goldenset_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    split = json.loads((DATA / "split_rag.json").read_text(encoding="utf-8"))
    ids = set(split["evolve"]) | set(split["heldout"])
    cases = [i for i in items if i["id"] in ids and i.get("chunks")]
    rows = [evaluate(v, cases) for v in VARIANTS]
    print(f"{'variant':<14}{'recall@8':>10}{'chunks':>8}{'avg_tok':>10}")
    for r in rows:
        print(f"{r['variant']:<14}{r['recall@8']:>10.3f}{r['chunks']:>8}{r['avg_tokens']:>10.1f}")
    out = ROOT / "eval" / "reports" / "chunking.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
