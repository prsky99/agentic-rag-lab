"""테넌트 스코프와 필터 (3.5절).

`TenantScope` 없이는 검색 함수를 호출할 수 없습니다. 필터는 qdrant 모델과 같은
모양의 최소 구현으로 만들 수 있어, qdrant 없이도 격리 테스트를 돌릴 수 있습니다.
"""
from __future__ import annotations

from dataclasses import dataclass

try:                                                  # 운영 경로
    from qdrant_client import models as m

    Filter, FieldCondition, MatchValue, MatchAny = (
        m.Filter, m.FieldCondition, m.MatchValue, m.MatchAny)
except ImportError:                                   # 테스트/로컬 경로
    from services.retriever._minimal_models import (
        FieldCondition, Filter, MatchAny, MatchValue)


@dataclass(frozen=True)
class TenantScope:
    """검색 권한의 유일한 표현. 이 객체 없이는 검색 함수를 호출할 수 없습니다."""

    tenant_id: str
    allowed_doc_ids: frozenset[str] | None = None   # 테넌트 내 추가 제한(선택)

    def __post_init__(self):
        if not self.tenant_id or self.tenant_id.strip() == "":
            raise ValueError("tenant_id is required")


def build_filter(scope: TenantScope):
    must = [FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id))]
    if scope.allowed_doc_ids is not None:
        must.append(FieldCondition(key="doc_id",
                                   match=MatchAny(any=list(scope.allowed_doc_ids))))
    return Filter(must=must)
