"""컨텍스트 조립 (8.1절).

검색이 준 문단은 그대로 붙이지 않고 순서·중복·길이를 정한 뒤, 각 문단에
인용 표식을 붙입니다. 여기서 정한 순서와 표식이 그대로 답변의 인용이 되므로,
인용 검증(8.4)과 채점(8.5)이 같은 좌표계를 씁니다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import tokens
from .retrieve import Passage

SENTENCE = re.compile(r"(?<=[.!?。])\s+|(?<=다\.)\s*")
MARKER = re.compile(r"^\[\d+\]\s*")


@dataclass
class Citation:
    marker: int
    doc_id: str
    chunk_id: str
    span: str


@dataclass
class Context:
    text: str
    citations: list[Citation] = field(default_factory=list)
    used: list[Passage] = field(default_factory=list)


def sentences(text: str) -> list[str]:
    """문장 단위로 나눕니다. 마크다운 줄바꿈은 문장 경계로 보지 않고 공백으로 취급합니다."""
    flat = re.sub(r"[ \t]*\n[ \t]*", " ", text or "")
    return [s.strip() for s in re.split(r"(?<=[.!?。])\s+", flat) if s and s.strip()]


def body_of(p) -> str:
    """청크 텍스트에서 절 제목을 뺀 본문. 제목은 답변 문장이 아니라 검색용입니다."""
    text = (p.text or "").strip()
    if p.section and text.startswith(p.section):
        return text[len(p.section):].strip()
    return text


def best_span(text: str, question: str, max_len: int = 100) -> str:
    """질문과 가장 겹치는 문장. 인용의 최소 단위입니다."""
    body = MARKER.sub("", text).strip()
    best, best_score = body[:max_len], -1.0
    q = tokens(question)
    for s in sentences(body):
        s = s.strip(" -•\t")
        if not s:
            continue
        t = tokens(s)
        score = (len(q & t) / len(q) if q else 0.0) + (0.2 if any(c.isdigit() for c in s) else 0.0)
        if score > best_score:
            best, best_score = s, score
    best = best.strip()
    if best not in body:
        best = body[:max_len]
    return best[:max_len]


def _jaccard(a: str, b: str) -> float:
    A, B = tokens(a), tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _order(passages: list[Passage], mode: str) -> list[Passage]:
    if mode == "relevance_asc":
        return sorted(passages, key=lambda p: p.score)
    if mode == "doc_order":
        return sorted(passages, key=lambda p: (p.doc_id, p.chunk_id))
    return sorted(passages, key=lambda p: -p.score)


def build_context(passages: list[Passage], question: str,
                  cfg: dict | None = None) -> Context:
    cfg = cfg or {}
    final_k = int(cfg.get("final_k", 8))
    max_chars = int(cfg.get("max_chars", 12000))
    dedup = float(cfg.get("dedup_threshold", 0.92))
    keep_tables = bool(cfg.get("keep_tables", True))

    kept: list[Passage] = []
    for p in _order(passages, str(cfg.get("order", "relevance_desc"))):
        if len(kept) >= final_k:
            break
        if any(_jaccard(p.text, k.text) >= dedup for k in kept):
            continue
        kept.append(p)

    # 가장 관련 높은 문단을 끝에 두는 옵션(긴 컨텍스트에서는 처음·끝 문단이 더 잘 쓰입니다)
    if cfg.get("order") == "relevance_desc" and len(kept) > 2:
        kept = kept[1:] + kept[:1]

    blocks: list[str] = []
    citations: list[Citation] = []
    used: list[Passage] = []
    budget = max_chars
    for i, p in enumerate(kept, start=1):
        body = p.text.strip()
        if not keep_tables and "|" in body:
            continue
        if len(body) > budget:
            body = body[: max(0, budget)]
        if not body:
            continue
        block = f"[{i}] ({p.doc_id} · {p.section})\n{body}"
        blocks.append(block)
        citations.append(Citation(i, p.doc_id, p.chunk_id, best_span(body_of(p), question)))
        used.append(p)
        budget -= len(block) + 2
        if budget <= 0:
            break

    return Context(text="\n\n".join(blocks), citations=citations, used=used)
