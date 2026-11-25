"""검색 도구: 하이브리드(BM25 + 벡터 대체) + RRF (3장, 8.1절).

``backend: local``이면 저장소에 들어 있는 코퍼스를 메모리에서 색인합니다.
``backend: http``이면 ``RETRIEVER_URL``의 검색 서비스가 Qdrant를 조회합니다.
로컬 모드의 "dense"는 문자 3-gram 해싱 벡터입니다. 임베딩 모델 없이도 융합·
리랭킹·게이트 배선을 그대로 시험하기 위한 대체이며, 품질 지표는 실제
임베딩 모델을 붙였을 때와 달라집니다.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from . import llm as llm_mod
from .llm import scoped_timeout

TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")
HEADING = re.compile(r"^##\s+(.*)$")
DIM = 256


@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    tenant: str
    title: str
    section: str
    text: str
    score: float = 0.0

    def as_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id,
                "tenant": self.tenant, "title": self.title,
                "section": self.section, "text": self.text, "score": self.score}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text or "") if len(t) > 1]


def parse_front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end < 0:
        return {}, raw
    block = raw[3:end].strip()
    meta = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, raw[end + 4:].lstrip("\n")


def load_corpus(corpus_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(Path(corpus_dir).rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        doc_id = meta.get("doc_id") or str(path.relative_to(corpus_dir).with_suffix(""))
        docs.append({"doc_id": doc_id, "tenant": meta.get("tenant", "acme"),
                     "title": meta.get("title", path.stem), "body": body,
                     "path": str(path)})
    return docs


def chunk_doc(doc: dict) -> list[Passage]:
    """제목(##) 단위로 자릅니다. 청크 ID는 ``<doc_id>#<i>``로 결정적입니다."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in doc["body"].splitlines():
        m = HEADING.match(line)
        if m:
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    out = []
    for i, (heading, lines) in enumerate(sections):
        body = "\n".join(lines).strip()
        if not body and not heading:
            continue
        text = (f"{heading}\n{body}" if heading else body).strip()
        if not text:
            continue
        out.append(Passage(chunk_id=f"{doc['doc_id']}#{i}", doc_id=doc["doc_id"],
                           tenant=doc["tenant"], title=doc["title"],
                           section=heading or doc["title"], text=text))
    return out


def hash_vector(text: str, dim: int = DIM) -> list[float]:
    """문자 3-gram 해싱 벡터(로컬 dense 대체)."""
    vec = [0.0] * dim
    clean = re.sub(r"\s+", " ", (text or "").lower())
    for i in range(len(clean) - 2):
        h = int(hashlib.md5(clean[i:i + 3].encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class LocalIndex:
    def __init__(self, passages: list[Passage], cfg: dict) -> None:
        self.passages = passages
        self.cfg = cfg or {}
        self.by_id = {p.chunk_id: p for p in passages}
        self._docs_tokens = [tokenize(p.text) for p in passages]
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        for toks in self._docs_tokens:
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = (sum(len(t) for t in self._docs_tokens) / len(passages)) if passages else 0.0
        self._vecs = [hash_vector(p.text) for p in passages]
        self.aliases = {k: set(v) for k, v in (self.cfg.get("tenant_aliases") or {}).items()}

    def _tenants(self, tenant: str) -> set[str]:
        return self.aliases.get(tenant, {tenant})

    def idf(self, term: str) -> float:
        """어휘 희소도. 추출형 생성이 흔한 단어 쪽으로 치우치지 않게 합니다."""
        n = len(self.passages)
        df = self._df.get(term)
        if df is None:
            cands = [d for t, d in self._df.items()
                     if len(t) >= 2 and len(term) >= 2
                     and (t[:2] == term[:2] or t in term or term in t)]
            df = min(cands) if cands else 0.5
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _bm25(self, query: str, cand: list[int], k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
        q = tokenize(query)
        n = len(self.passages)
        scores = []
        for i in cand:
            tf = self._tf[i]
            dl = len(self._docs_tokens[i]) or 1
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self._df.get(t, 0) + 0.5) / (self._df.get(t, 0) + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / (self._avgdl or 1)))
            scores.append((i, s))
        return scores

    def search(self, question: str, tenant: str, top_k: int = 40) -> list[Passage]:
        allowed = self._tenants(tenant)
        cand = [i for i, p in enumerate(self.passages) if p.tenant in allowed]
        if not cand:
            return []
        sparse = sorted(self._bm25(question, cand), key=lambda x: -x[1])
        qv = hash_vector(question)
        dense = sorted(((i, cosine(qv, self._vecs[i])) for i in cand), key=lambda x: -x[1])

        sparse_ids = [self.passages[i].chunk_id for i, s in sparse[:top_k] if s > 0]
        dense_ids = [self.passages[i].chunk_id for i, s in dense[:top_k] if s > 0]
        fused = rrf([sparse_ids, dense_ids], k=int(self.cfg.get("rrf_k", 60)),
                    weights=[float(self.cfg.get("bm25_weight", 0.45)),
                             float(self.cfg.get("vector_weight", 0.55))])
        out = []
        for chunk_id, score in fused[:top_k]:
            p = self.by_id[chunk_id]
            out.append(Passage(**{**p.__dict__, "score": round(score, 6)}))
        return out


class RetrievalUnavailable(RuntimeError):
    """검색 서비스 응답 없음(연결 실패·시간 초과·오류 응답·형식 오류)."""


class HttpRetriever:
    """검색 서비스(services/retriever)를 호출하는 백엔드."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def search(self, question: str, tenant: str, top_k: int = 40) -> list[Passage]:
        # 파이프라인이 남은 시간을 넘기면 그 시간이 요청 전체(본문 수신 포함)의 상한이 됩니다.
        timeout = scoped_timeout(5.0)
        try:
            body = llm_mod._post_json(
                self.base_url + "/search",
                {"query": question, "tenant": tenant, "top_k": top_k},
                {llm_mod.DEADLINE_HEADER: str(int(timeout * 1000))}, timeout)
            return [Passage(chunk_id=r["chunk_id"], doc_id=r.get("doc_id", ""),
                            tenant=r.get("tenant", tenant), title=r.get("title", ""),
                            section=r.get("section", ""), text=r.get("text", ""),
                            score=float(r.get("score", 0.0)))
                    for r in body.get("results", [])]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            # TimeoutError·URLError는 OSError입니다. 파이프라인이 통제된 기권으로 바꿉니다.
            raise RetrievalUnavailable(f"{type(exc).__name__}: {exc}") from exc


def build_retriever(cfg: dict, config_dir: Path):
    backend = str(cfg.get("backend", "local")).lower()
    if backend == "auto":
        backend = "http" if (cfg.get("service_url") or os.environ.get("RETRIEVER_URL")) else "local"
    if backend == "http":
        base = cfg.get("service_url") or os.environ.get("RETRIEVER_URL", "")
        if not base:
            raise RuntimeError("retrieval.backend=http이지만 service_url/RETRIEVER_URL이 비어 있습니다")
        return HttpRetriever(base)
    corpus = Path(cfg.get("corpus_dir", "../data/corpus"))
    if not corpus.is_absolute():
        corpus = (config_dir / corpus).resolve()
    passages = []
    for doc in load_corpus(corpus):
        passages.extend(chunk_doc(doc))
    return LocalIndex(passages, cfg)


def lexical_score(question: str, passage: Passage) -> float:
    q = set(tokenize(question))
    p = set(tokenize(passage.text))
    if not q:
        return 0.0
    cov = len(q & p) / len(q)
    nums = {t for t in q if any(ch.isdigit() for ch in t)}
    bonus = 0.15 if nums and nums <= p else 0.0
    return round(cov + bonus + min(0.1, passage.score), 6)


_CROSS_ENCODERS: dict[str, object] = {}
_CE_LOCK = threading.Lock()


def _cross_encoder(name: str):
    """교차 인코더를 프로세스당 한 번만 읽습니다. 읽지 못하면 None을 기억해 재시도를 막습니다.

    허브 저장소 이름(``BAAI/bge-reranker-v2-m3``)이나 로컬 경로일 때만 읽습니다. 조직 이름이
    없는 이름은 허브를 조회하지 않고 어휘 점수를 씁니다.
    """
    with _CE_LOCK:
        if name not in _CROSS_ENCODERS:
            model = None
            if "/" in name or Path(name).exists():
                try:
                    from sentence_transformers import CrossEncoder  # type: ignore
                    model = CrossEncoder(name, max_length=512)
                except Exception:                # noqa: BLE001 - 모델 없음/로드 실패
                    model = None
            _CROSS_ENCODERS[name] = model
        return _CROSS_ENCODERS[name]


def rerank(question: str, passages: list[Passage], top_n: int,
           cfg: dict | None = None) -> list[Passage]:
    """교차 인코더가 있으면 쓰고, 없으면 어휘 점수로 정렬합니다."""
    cfg = cfg or {}
    if cfg.get("enabled", True) and cfg.get("model"):
        try:
            model = _cross_encoder(str(cfg["model"]))
            if model is None:
                raise LookupError(cfg["model"])
            pairs = [(question, p.text[:2000]) for p in passages]
            scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
            ranked = sorted(((p, float(s)) for p, s in zip(passages, scores)),
                            key=lambda x: -x[1])
            return [Passage(**{**p.__dict__, "score": round(s, 6)}) for p, s in ranked[:top_n]]
        except Exception:                        # noqa: BLE001 - 모델 없음/로드 실패
            pass
    ranked = sorted(((p, lexical_score(question, p)) for p in passages), key=lambda x: -x[1])
    return [Passage(**{**p.__dict__, "score": s}) for p, s in ranked[:top_n]]
