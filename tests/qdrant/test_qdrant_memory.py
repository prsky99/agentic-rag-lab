"""Qdrant 적재·검색 시험 (3.2절, 3.4절).

qdrant-client의 메모리 모드(``QdrantClient(":memory:")``)로 실제 업서트와 하이브리드
검색을 돌립니다. 임베딩 모델(fastembed) 대신 단어 해시 기반의 가짜 임베더를 씁니다.
qdrant-client가 설치되어 있지 않으면 건너뜁니다(requirements.txt에 1.20.0 고정).
"""
from __future__ import annotations

import re
import uuid
import zlib

import pytest

pytest.importorskip("qdrant_client")
np = pytest.importorskip("numpy")                    # qdrant-client 의존성

from qdrant_client import QdrantClient  # noqa: E402

from eval.index import load_to_qdrant  # noqa: E402
from services.retriever.index import point_id  # noqa: E402
from services.retriever.scope import TenantScope  # noqa: E402
from services.retriever.search import hybrid_search  # noqa: E402

DIM = 32
WORD = re.compile(r"[0-9A-Za-z가-힣]+")


def _bucket(w: str, n: int) -> int:
    return zlib.crc32(w.encode("utf-8")) % n


class FakeDense:
    def _one(self, text: str):
        v = [0.0] * DIM
        for w in WORD.findall(text):
            v[_bucket(w, DIM)] += 1.0
        return np.array(v if any(v) else [1.0] + [0.0] * (DIM - 1))   # fastembed처럼 ndarray

    def embed(self, texts, batch_size=32):
        return (self._one(t) for t in texts)

    def query_embed(self, text):
        return iter([self._one(text)])


class _Sparse:
    def __init__(self, indices, values):
        self.indices, self.values = indices, values


class FakeSparse:
    def _one(self, text: str) -> _Sparse:
        counts: dict[int, float] = {}
        for w in WORD.findall(text):
            k = _bucket(w, 100_000)
            counts[k] = counts.get(k, 0.0) + 1.0
        idx = sorted(counts)
        return _Sparse(np.array(idx), np.array([counts[i] for i in idx]))

    def embed(self, texts, batch_size=32):
        return (self._one(t) for t in texts)

    def query_embed(self, text):
        return iter([self._one(text)])


PASSAGES = [
    {"chunk_id": "acme/refund-policy#1", "doc_id": "acme/refund-policy", "tenant": "acme",
     "title": "환불", "section": "기한", "text": "환불은 구매 후 14일 이내에 요청할 수 있습니다."},
    {"chunk_id": "acme/refund-policy#2", "doc_id": "acme/refund-policy", "tenant": "acme",
     "title": "환불", "section": "예외", "text": "디지털 상품은 다운로드 후 환불되지 않습니다."},
]
OTHER = [
    {"chunk_id": "globex/secret#0", "doc_id": "globex/secret", "tenant": "globex",
     "title": "청룡", "section": "예산", "text": "청룡-7731 프로젝트 환불 예산은 12억 원입니다."},
]


def test_point_id_is_deterministic_uuid():
    a = point_id("acme/refund-policy#1")
    assert a == point_id("acme/refund-policy#1")
    assert a != point_id("acme/refund-policy#2")
    assert str(uuid.UUID(a)) == a


def test_upsert_and_hybrid_search_return_logical_chunk_id():
    client = QdrantClient(":memory:")
    dense, sparse = FakeDense(), FakeSparse()
    assert load_to_qdrant(client, PASSAGES, "demo", "idx-test", "docs_current",
                          dense, sparse) == 2
    assert load_to_qdrant(client, OTHER, "other", "idx-test", "docs_current",
                          dense, sparse) == 1
    # 같은 청크를 다시 적재해도 점이 늘지 않습니다(결정적 ID → 덮어쓰기)
    load_to_qdrant(client, PASSAGES, "demo", "idx-test", "docs_current", dense, sparse)
    assert client.count("idx-test").count == 3

    hits = hybrid_search(client, "docs_current", TenantScope("demo"), "환불 14일 이내",
                         dense, sparse, top_k=5)
    ids = [h["chunk_id"] for h in hits]
    assert ids and ids[0] == "acme/refund-policy#1"
    assert set(ids) <= {"acme/refund-policy#1", "acme/refund-policy#2"}   # globex 누출 없음
    assert hits[0]["point_id"] == point_id("acme/refund-policy#1")
    assert hits[0]["tenant_id"] == "demo" and hits[0]["source_tenant"] == "acme"


def test_alias_moves_to_new_snapshot():
    client = QdrantClient(":memory:")
    dense, sparse = FakeDense(), FakeSparse()
    load_to_qdrant(client, PASSAGES[:1], "demo", "idx-a", "docs_current", dense, sparse)
    load_to_qdrant(client, PASSAGES, "demo", "idx-b", "docs_current", dense, sparse)
    aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    assert aliases["docs_current"] == "idx-b"
