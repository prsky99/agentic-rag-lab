"""검색 진입점 (3.5절).

모든 검색은 이 파일의 `search` 하나를 지납니다. `scope`는 기본값이 없습니다.
백엔드는 Qdrant가 있으면 Qdrant, 없으면 스냅샷 파일을 메모리에서 검색합니다.
어느 쪽이든 필터는 같은 `build_filter` 결과를 씁니다.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "domains" / "rag"))

from services.retriever.filtering import matches
from services.retriever.scope import TenantScope, build_filter

TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")

try:
    from qdrant_client import QdrantClient
except ImportError:                                   # 로컬/테스트
    QdrantClient = None                                # type: ignore

QDRANT_URL = os.environ.get("QDRANT_URL", "")
BACKEND = os.environ.get("RETRIEVER_BACKEND", "").lower() or ("qdrant" if QDRANT_URL else "local")
COLLECTION = os.environ.get("COLLECTION_ALIAS", "docs_current")   # 별칭. 3.3절의 재색인으로 대상이 바뀝니다
SNAPSHOT_FILE = os.environ.get("INDEX_SNAPSHOT_FILE", "")
CORPUS_DIR = os.environ.get("CORPUS_DIR", str(Path("/app/domains/rag/data/corpus")))
TENANT_ALIASES = {k: set(v) for k, v in
                  json.loads(os.environ.get("TENANT_ALIASES", "{}")).items()}
_client = None
_payloads: list[dict] | None = None


def _client_or_none():
    global _client
    if BACKEND != "qdrant":
        return None
    if _client is None and QDRANT_URL and QdrantClient is not None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(text or "") if len(t) > 1}


def _hit(t: str, doc: set[str]) -> bool:
    """한국어 조사 차이를 흡수한 어휘 매칭."""
    if t in doc:
        return True
    if len(t) < 2:
        return False
    return any(len(w) >= 2 and (t in w or w in t or t[:2] == w[:2]) for w in doc)


def load_payloads() -> list[dict]:
    global _payloads
    if _payloads is not None:
        return _payloads
    path = Path(SNAPSHOT_FILE) if SNAPSHOT_FILE else None
    if path and path.exists():
        _payloads = json.loads(path.read_text(encoding="utf-8")).get("passages", [])
    elif Path(CORPUS_DIR).exists():
        # 스냅샷이 없으면 번들 코퍼스로 그 자리에서 만듭니다(개발·kind 기본 경로).
        from harness.retrieve import chunk_doc, load_corpus
        _payloads = [p.as_dict() | {"tenant_id": p.tenant}
                     for doc in load_corpus(Path(CORPUS_DIR)) for p in chunk_doc(doc)]
    else:
        _payloads = []
    return _payloads


def _aliases(tenant: str) -> set[str]:
    return TENANT_ALIASES.get(tenant, {tenant})


def local_search(payloads: list[dict], scope: TenantScope, query: str,
                 top_k: int = 50) -> list[dict]:
    """메모리 검색. 필터를 먼저 적용하고 그 다음에 점수를 매깁니다."""
    flt = build_filter(scope)
    q = _tokens(query)
    scored = []
    allowed = _aliases(scope.tenant_id)
    for p in payloads:
        payload = {"tenant_id": p.get("tenant_id") or p.get("tenant"),
                   "doc_id": p.get("doc_id"), "chunk_id": p.get("chunk_id"),
                   "text": p.get("text", ""), "title": p.get("title", ""),
                   "section": p.get("section", "")}
        payload_scope = {**payload, "tenant_id": scope.tenant_id} \
            if payload["tenant_id"] in allowed else payload
        if not matches(payload_scope, flt):
            continue
        text = _tokens(payload["text"])
        if not q or not text:
            continue
        score = sum(1 for t in q if _hit(t, text)) / len(q)
        if score > 0:
            scored.append({**payload, "score": round(score, 6)})
    scored.sort(key=lambda r: -r["score"])
    return scored[:top_k]


_MODEL_LOCK = threading.Lock()


def warm() -> bool:
    """Qdrant 백엔드이면 임베딩 모델을 미리 읽습니다(준비 상태 검사용). 로컬 백엔드는 할 일이 없습니다.

    모델을 처음 읽을 때는 내려받기·초기화에 수 초에서 수십 초가 걸립니다. 첫 검색 요청이 그
    시간을 떠안지 않도록 ``/ready``가 이 함수를 부릅니다. 잠금으로 동시에 두 번 읽지 않습니다.
    """
    if BACKEND != "qdrant":
        return True
    with _MODEL_LOCK:
        _models()
    return True


@lru_cache(maxsize=1)
def _models():
    """임베딩 모델은 무겁습니다. 요청마다 만들지 않고 첫 호출에서 한 번만 만듭니다.

    모델 이름·차원은 색인기와 같은 ``services/retriever/embedding.py``에서 읽습니다.
    """
    try:
        from services.retriever.embedding import load_models
        return load_models()
    except ImportError as exc:                        # pragma: no cover
        raise RuntimeError("qdrant backend requires qdrant-client and fastembed") from exc


@lru_cache(maxsize=8)
def _check_dim(collection: str) -> None:
    """컬렉션의 dense 차원이 설정과 같은지 한 번 확인합니다(다르면 색인과 검색 모델이 다릅니다)."""
    from services.retriever.embedding import embed_dim
    from services.retriever.index import dense_size
    have = dense_size(_client_or_none(), collection)
    if have is not None and have != embed_dim():
        raise RuntimeError(f"컬렉션 {collection}의 dense 차원 {have}이 임베딩 차원 "
                           f"{embed_dim()}과 다릅니다. 같은 EMBED_MODEL로 다시 색인해 주세요")


def search(scope: TenantScope, query: str, top_k: int = 50) -> list[dict]:
    """모든 검색 진입점은 이 함수 하나를 지납니다. scope는 기본값이 없습니다."""
    return _qdrant_query(scope, build_filter(scope), query, top_k)


def _qdrant_query(scope: TenantScope, query_filter, query: str,
                  top_k: int) -> list[dict]:
    """실제 백엔드 호출. 테스트에서는 이 함수를 몽키패치해 필터 전달을 검사합니다."""
    client = _client_or_none()
    if client is None:
        return local_search(load_payloads(), scope, query, top_k)
    from services.retriever.search import hybrid_search
    dense, sparse = _models()                         # 프로세스당 한 번만 만듭니다
    _check_dim(COLLECTION)
    return hybrid_search(client, COLLECTION, scope, query, dense, sparse,
                         top_k=top_k, query_filter=query_filter)
