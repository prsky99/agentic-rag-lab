"""컬렉션 생성·임베딩·적재 (3.2절, 3.4절).

payload 인덱스가 없으면 테넌트 필터가 풀스캔이 됩니다. 멀티테넌시의 전제입니다.

Qdrant의 point ID는 부호 없는 정수나 UUID만 받습니다. ``acme/refund-policy#1`` 같은
논리 청크 ID를 그대로 넣으면 서버와 메모리 백엔드 모두 오류를 냅니다. 그래서 물리 ID는
``point_id()``가 고정 네임스페이스의 uuid5로 만들고, 논리 ID(chunk_id)는 payload에 보존합니다.
같은 chunk_id는 언제나 같은 UUID가 되므로 다시 색인해도 점이 중복되지 않고 덮어써집니다.
"""
from __future__ import annotations

import uuid
from typing import Iterable

from qdrant_client import QdrantClient, models

from services.retriever.embedding import embed_dim

# 모델 이름과 차원은 services/retriever/embedding.py에서만 정합니다(EMBED_MODEL·EMBED_DIM).

# 값을 바꾸면 모든 point ID가 바뀌므로 바꾸지 않습니다(재색인 전체가 필요해집니다).
POINT_NAMESPACE = uuid.UUID("6f1c2a8e-3b5d-5e7a-9c41-2d8b7e0f4a13")


def point_id(chunk_id: str) -> str:
    """논리 chunk_id → 결정적 물리 point ID(UUID 문자열)."""
    if not chunk_id:
        raise ValueError("chunk_id is required")
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def dense_size(client: QdrantClient, name: str) -> int | None:
    """컬렉션(또는 별칭)의 dense 벡터 차원. 읽지 못하면 None입니다."""
    try:
        vectors = client.get_collection(name).config.params.vectors
        return int(vectors["dense"].size)
    except Exception:                                 # noqa: BLE001 - 없는 컬렉션·다른 구성
        return None


def ensure_collection(client: QdrantClient, name: str, dim: int | None = None) -> None:
    """컬렉션이 없으면 만듭니다. ``dim``을 비우면 설정의 차원(``embed_dim()``)을 씁니다.

    이미 있는 컬렉션의 차원이 다르면 오류를 냅니다. 모델을 바꿨다면 새 컬렉션에 다시
    색인하고 별칭을 옮겨야 합니다(3.3절).
    """
    dim = embed_dim() if dim is None else int(dim)
    if client.collection_exists(name):
        have = dense_size(client, name)
        if have is not None and have != dim:
            raise ValueError(f"컬렉션 {name}의 dense 차원은 {have}인데 임베딩 차원은 {dim}입니다. "
                             "새 컬렉션에 다시 색인해 주세요")
        return
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(size=dim,
                                                     distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF)},
    )
    # 멀티테넌시의 전제: payload 인덱스가 없으면 필터가 풀스캔이 됩니다
    # (메모리 모드는 payload 인덱스를 무시하고 경고만 냅니다. 서버 Qdrant에서는 필요합니다)
    client.create_payload_index(name, "tenant_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "doc_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "chunk_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "lang", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "updated_at", models.PayloadSchemaType.DATETIME)


def _floats(v) -> list[float]:
    return v.tolist() if hasattr(v, "tolist") else [float(x) for x in v]


def _ints(v) -> list[int]:
    return v.tolist() if hasattr(v, "tolist") else [int(x) for x in v]


def embed_all(chunks: list[dict], dense, sparse):
    """dense·sparse는 fastembed의 TextEmbedding·SparseTextEmbedding 인스턴스입니다."""
    texts = [c["text"] for c in chunks]
    dense_vecs = list(dense.embed(texts, batch_size=32))
    sparse_vecs = list(sparse.embed(texts, batch_size=32))
    points = []
    for c, dv, sv in zip(chunks, dense_vecs, sparse_vecs):
        points.append(models.PointStruct(
            id=point_id(c["chunk_id"]),               # 물리 ID: 결정적 UUID
            vector={
                "dense": _floats(dv),
                "bm25": models.SparseVector(indices=_ints(sv.indices),
                                            values=_floats(sv.values)),
            },
            payload={
                "chunk_id": c["chunk_id"],            # 논리 ID: 검색 결과는 이 값을 돌려줍니다
                "tenant_id": c.get("tenant_id") or c.get("tenant"),
                "source_tenant": c.get("source_tenant") or c.get("tenant"),
                "doc_id": c["doc_id"],
                "title": c.get("title", ""),
                "section": c.get("section", ""),
                "lang": c.get("lang", "ko"),
                "text": c["text"],
                "updated_at": c.get("updated_at"),
            },
        ))
    return points


def _batches(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def upsert_chunks(client: QdrantClient, collection: str, chunks: list[dict],
                  dense, sparse, batch_size: int = 64, dim: int | None = None) -> int:
    """청크를 임베딩해 업서트합니다.

    ``dim``은 기대하는 dense 차원입니다. 색인 CLI는 설정의 ``embed_dim()``을 넘기고, 모델이
    만든 벡터의 길이가 그 값과 다르면 적재하지 않습니다. 비우면 첫 벡터의 길이로 컬렉션을
    만듭니다(가짜 임베딩을 쓰는 시험용).
    """
    if not chunks:
        return 0
    n = 0
    for part in _batches(chunks, batch_size):
        points = embed_all(part, dense, sparse)
        got = len(points[0].vector["dense"])
        if dim is not None and got != int(dim):
            raise ValueError(f"임베딩 차원 {got}이 설정 차원 {dim}과 다릅니다"
                             "(EMBED_MODEL·EMBED_DIM을 확인해 주세요)")
        ensure_collection(client, collection, dim=got)
        client.upsert(collection_name=collection, points=points, wait=True)
        n += len(points)
    return n


def point_alias(client: QdrantClient, alias: str, collection: str) -> None:
    """별칭이 있으면 원자 교체, 없으면 새로 만듭니다(3.3절)."""
    from services.retriever.reindex import create_alias, swap_alias
    names = {a.alias_name for a in client.get_aliases().aliases}
    if alias in names:
        swap_alias(client, alias, collection)
    else:
        create_alias(client, alias, collection)
