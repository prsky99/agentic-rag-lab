"""테넌트 스코프 저장소 (11.2절).

저장소 접근 객체가 테넌트를 들고 있으면, 호출자는 테넌트를 잊을 수 없습니다.
필터를 호출마다 넘기는 방식은 한 번 빠뜨리면 조용히 전체를 반환합니다.

``search``가 받는 ``vector``는 색인과 같은 모델(``services/retriever/embedding.py``의
``EMBED_MODEL``, 차원 ``EMBED_DIM``)로 만든 dense 벡터여야 합니다.
"""
from __future__ import annotations

try:
    from qdrant_client import models
except ImportError:                                   # 로컬/테스트
    from services.retriever import _minimal_models as models  # type: ignore


class TenantScopedStore:
    def __init__(self, client, tenant_id: str):
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id is required")
        self._c = client
        self._tenant = tenant_id

    def search(self, vector, k: int = 40) -> list:
        flt = models.Filter(must=[
            models.FieldCondition(key="tenant_id",
                                  match=models.MatchValue(value=self._tenant)),
            models.FieldCondition(key="deleted", match=models.MatchValue(value=False)),
        ])
        # qdrant-client 1.20에는 search()가 없으므로 query_points()를 씁니다.
        res = self._c.query_points(collection_name="docs", query=vector, using="dense",
                                   query_filter=flt, limit=k, with_payload=True)
        return res.points
