"""수집 검증 (3.5절).

문서 ID에 테넌트 접두를 강제합니다. 교차 테넌트 ID 충돌을 막고, 필터가 빠져도
ID만 보고 다른 테넌트 문서임을 알 수 있게 합니다.
"""
from __future__ import annotations

REQUIRED = {"tenant_id", "doc_id", "chunk_id", "text"}


def validate_chunk(c: dict) -> None:
    missing = REQUIRED - c.keys()
    if missing:
        raise ValueError(f"chunk missing fields: {sorted(missing)}")
    if not c["tenant_id"] or c["tenant_id"] != c["tenant_id"].strip():
        raise ValueError("invalid tenant_id")
    # 문서 ID에 테넌트 접두를 강제하면 교차 테넌트 ID 충돌도 막힙니다.
    # 구분자는 ':'(acme:doc-1) 또는 '/'(동봉 코퍼스의 acme/refund-policy)
    if not c["doc_id"].startswith((f"{c['tenant_id']}:", f"{c['tenant_id']}/")):
        raise ValueError("doc_id must be prefixed with tenant_id")
