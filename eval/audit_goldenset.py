#!/usr/bin/env python3
"""골든셋 감사 (9.2절).

네 가지를 확인합니다.
1. 분할 간 질문 중복: 정규화한 질문이 완전히 같거나(짧은 질문 포함), 5단어 묶음 자카드가 0.6 이상
2. 분할 간 정답 청크 겹침: evolve 문항의 정답 청크를 heldout·ood_docs 문항이 공유하는가
   (ood_intent는 같은 문서에 새 의도의 질문을 내는 분할이라 검사에서 뺍니다)
3. 정답의 개체(숫자·날짜)가 근거 청크에 실제로 있는가
4. 정답 가능/불가 표시와 근거 유무가 맞는가

    python eval/audit_goldenset.py --out eval/goldenset/audit_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "domains" / "rag" / "data"
WORD = re.compile(r"[0-9A-Za-z가-힣]+")


def ngrams(s: str, n: int = 5) -> set[tuple]:
    w = WORD.findall((s or "").lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def norm(s: str) -> str:
    """대소문자·공백·문장 부호를 지운 질문(짧은 질문의 완전 일치 검사용)."""
    return "".join(WORD.findall((s or "").lower()))


def jaccard(a: str, b: str) -> float:
    A, B = ngrams(a), ngrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def entities(text: str) -> set[str]:
    """숫자·날짜·영문 고유명사. 정답이 근거에 있는지 확인할 최소 단위."""
    out = set(re.findall(r"\d[\d,]*", text or ""))
    out |= {t for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or "")
            if t.lower() not in ("http", "https")}
    return out


def audit(items: list[dict]) -> list[dict]:
    found: list[dict] = []
    for i, a in enumerate(items):                       # 1. 분할 간 질문 중복
        for b in items[i + 1:]:
            if a["split"] == b["split"]:
                continue
            if norm(a["question"]) and norm(a["question"]) == norm(b["question"]):
                found.append({"kind": "question_duplicate", "a": a["id"], "b": b["id"]})
            elif jaccard(a["question"], b["question"]) >= 0.6:
                found.append({"kind": "question_overlap", "a": a["id"], "b": b["id"],
                              "jaccard": round(jaccard(a["question"], b["question"]), 3)})

    evolve_chunks = {c.get("chunk_id") for a in items if a["split"] == "evolve" for c in a.get("chunks") or []}
    for a in items:                                     # 2. 분할 간 정답 청크 겹침
        # ood_intent는 "같은 문서에 새로운 의도의 질문"을 내는 분할이라 청크 공유가 설계입니다.
        if a["split"] in ("heldout", "ood_docs"):
            shared = sorted({c.get("chunk_id") for c in a.get("chunks") or []} & evolve_chunks)
            if shared:
                found.append({"kind": "chunk_shared_with_evolve", "id": a["id"], "chunks": shared})

    for a in items:
        text = " ".join(c.get("text", "") for c in a.get("chunks") or [])
        if a.get("answerable"):
            if not a.get("chunks"):
                found.append({"kind": "answerable_without_chunk", "id": a["id"]})
                continue
            for ent in entities(a.get("reference", "")):    # 3. 숫자·날짜·영문 고유명사
                if ent and ent not in text:
                    found.append({"kind": "reference_not_supported", "id": a["id"],
                                  "entity": ent})
        else:                                           # 4. 간이 버전: 기권 문항에 근거가 붙었나
            if a.get("chunks"):
                found.append({"kind": "unanswerable_with_chunk", "id": a["id"]})
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldenset", default=str(DATA / "goldenset_v1.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "goldenset" / "audit_v1.jsonl"))
    a = ap.parse_args()

    items = [json.loads(line) for line in Path(a.goldenset).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    found = audit(items)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in found),
                   encoding="utf-8")
    print(f"[audit] {len(items)}문항, 문제 {len(found)}건 → {out}")
    for f in found[:10]:
        print("  -", f)
    return 0 if not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
