"""교차 인코더 리랭커 (3.4절).

50 → 8로 줄입니다. 자르는 이유: 판정에 필요한 것은 근거 문장이지 문단 전체가
아닙니다. 다만 잘라서 점수를 매기면 문맥이 사라지므로 2000자까지만 자릅니다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Ranked:
    chunk_id: str
    text: str
    score: float


class CrossEncoder:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 batch_size: int = 8, max_length: int = 512):
        from sentence_transformers import CrossEncoder as CE
        self._m = CE(model_name, max_length=max_length)
        self._bs = batch_size

    def rank(self, query: str, docs: list[dict], top_n: int = 8) -> list[Ranked]:
        pairs = [(query, d["text"][:2000]) for d in docs]   # 판정에 필요한 것은 근거 문장
        scores = self._m.predict(pairs, batch_size=self._bs, show_progress_bar=False)
        ranked = sorted(
            (Ranked(d["chunk_id"], d["text"], float(s)) for d, s in zip(docs, scores)),
            key=lambda r: -r.score,
        )
        return ranked[:top_n]


def lexical_rank(query: str, docs: list[dict], top_n: int = 8) -> list[Ranked]:
    """모델이 없을 때의 대체 정렬. 어휘 겹침만 봅니다."""
    q = {t.lower() for t in query.split() if len(t) > 1}
    out = []
    for d in docs:
        text = {t.lower() for t in str(d.get("text", "")).split() if len(t) > 1}
        score = len(q & text) / len(q) if q else 0.0
        out.append(Ranked(d["chunk_id"], d.get("text", ""), score))
    out.sort(key=lambda r: -r.score)
    return out[:top_n]
