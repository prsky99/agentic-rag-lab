"""일부러 새는 구현들 (3.6절 9~12번).

메타 테스트의 질문은 하나입니다: "우리 격리 테스트가 실제로 누출을 잡는가?"
아래 네 가지를 넣었을 때 격리 테스트가 실패하지 않는다면, 그 테스트는
통과해도 의미가 없습니다.

각 항목은 `services.retriever.service`의 부품 하나만 바꿉니다.
"""
from __future__ import annotations

from contextlib import contextmanager

from services.retriever import service
from services.retriever.scope import TenantScope, build_filter


def _none_filter(scope):
    """필터를 통째로 빠뜨린 구현."""
    return None


def _wrong_tenant_filter(scope):
    """테넌트를 잘못 넣은 구현(설정 오타·복사 실수)."""
    return build_filter(TenantScope("globex"))


def _always_match(payload, flt):
    """필터는 만들지만 적용하지 않는 구현."""
    return True


def _filter_without_allowlist(scope):
    """allowed_doc_ids 제한을 빠뜨린 구현."""
    f = build_filter(scope)
    f.must = [c for c in f.must if c.key != "doc_id"]
    return f


LEAKS = {
    "drop_filter": _none_filter,
    "drop_prefetch_filter": _none_filter,
    "wrong_tenant": _wrong_tenant_filter,
    "no_payload_check": _always_match,
    "drop_allowlist": _filter_without_allowlist,
}


@contextmanager
def leaky(name: str):
    """이름에 해당하는 누출을 잠시 켭니다."""
    fn = LEAKS[name]
    old_build = service.build_filter
    old_matches = service.matches
    try:
        if name == "no_payload_check":
            service.matches = fn
        else:
            service.build_filter = fn
        yield
    finally:
        service.build_filter = old_build
        service.matches = old_matches


def isolation_violations(payloads: list[dict], scope: TenantScope,
                         query: str = "계약") -> list[str]:
    """격리 규칙 위반 목록. 비어 있으면 격리가 지켜진 것입니다."""
    out = []
    results = service.local_search(payloads, scope, query, top_k=50)
    for r in results:
        if r["tenant_id"] != scope.tenant_id:
            out.append(f"foreign tenant {r['tenant_id']} in {r['chunk_id']}")
        if scope.allowed_doc_ids is not None and r["doc_id"] not in scope.allowed_doc_ids:
            out.append(f"doc outside allowlist {r['doc_id']}")
    return out
