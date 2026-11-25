"""Qdrant 하이브리드 검색 (3.2절).

prefetch와 최종 질의에 **같은 필터**를 씁니다. 한쪽만 걸면 융합 전에 다른
테넌트 문서가 섞이고, 융합 점수는 이미 오염된 뒤입니다.

``dense_model``·``sparse_model``은 색인에 쓴 것과 같은 모델이어야 합니다. 서비스와 색인
CLI는 둘 다 ``services/retriever/embedding.py``의 ``load_models()``로 만듭니다.
"""
from __future__ import annotations

from qdrant_client import QdrantClient, models

from services.retriever.scope import TenantScope, build_filter


def hybrid_search(
    client: QdrantClient,
    collection: str,
    scope: TenantScope,          # 테넌트는 기본값 없는 필수 인자입니다
    query: str,
    dense_model,
    sparse_model,
    top_k: int = 50,
    candidate_k: int = 50,
    query_filter=None,           # 호출자가 만든 필터를 그대로 씁니다(없으면 여기서 만듭니다)
) -> list[dict]:
    dense_vec = _floats(next(iter(dense_model.query_embed(query))))
    sparse_vec = next(iter(sparse_model.query_embed(query)))

    # prefetch와 최종 질의에 같은 필터를 씁니다. 한쪽만 걸면 융합 전에 섞입니다.
    tenant_filter = query_filter or build_filter(scope)
    res = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=candidate_k,
                            filter=tenant_filter),
            models.Prefetch(
                query=models.SparseVector(indices=_floats(sparse_vec.indices, int),
                                          values=_floats(sparse_vec.values)),
                using="bm25", limit=candidate_k, filter=tenant_filter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        query_filter=tenant_filter,
        with_payload=True,
    )
    return [to_result(p) for p in res.points]


def _floats(v, cast=float) -> list:
    return v.tolist() if hasattr(v, "tolist") else [cast(x) for x in v]


def to_result(p) -> dict:
    """point → 검색 결과. 물리 ID(UUID)가 아니라 payload의 논리 chunk_id를 돌려줍니다."""
    payload = dict(p.payload or {})
    return {**payload, "chunk_id": payload.get("chunk_id") or str(p.id),
            "point_id": str(p.id), "score": p.score}
