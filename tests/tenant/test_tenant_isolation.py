"""테넌트 격리 테스트 12건 (3.5절, 11.2절).

1~8은 규칙을 검사하고, 9~12는 그 규칙 검사가 실제로 누출을 잡는지 검사합니다.
"""
from __future__ import annotations

import pytest

from services.retriever import service
from services.retriever.filtering import matches
from services.retriever.scope import TenantScope, build_filter
from tests.meta.leaky_retriever import isolation_violations, leaky

ACME = TenantScope("acme")
GLOBEX = TenantScope("globex")
CANARY = "청룡-7731"            # globex 문서에만 존재


def ids(payloads, scope, q="계약"):
    return {r["tenant_id"] for r in service.local_search(payloads, scope, q, top_k=50)}


# 1. 기본 검색은 자기 테넌트만 반환합니다
def test_only_own_tenant(corpus_payloads):
    assert ids(corpus_payloads, ACME) <= {"acme"}


# 2. 결과 payload에는 자기 테넌트만 있습니다
def test_no_foreign_payload(corpus_payloads):
    results = service.local_search(corpus_payloads, ACME, "계약 예산", 50)
    assert all(r["tenant_id"] == "acme" for r in results)
    assert CANARY not in str(results)


# 3. 빈 테넌트는 거부됩니다
def test_empty_tenant_rejected():
    with pytest.raises(ValueError):
        TenantScope("")


# 4. 공백 테넌트는 거부됩니다
def test_blank_tenant_rejected():
    with pytest.raises(ValueError):
        TenantScope("   ")


# 5. allowed_doc_ids 제한이 실제로 적용됩니다
def test_doc_allowlist(corpus_payloads):
    scope = TenantScope("acme", frozenset({"acme:doc-1"}))
    got = {r["doc_id"] for r in service.local_search(corpus_payloads, scope, "계약", 50)}
    assert got <= {"acme:doc-1"}


# 6. 허용 목록에 다른 테넌트 문서를 넣어도 결과에서 제외됩니다
def test_allowlist_cannot_cross_tenant(corpus_payloads):
    scope = TenantScope("acme", frozenset({"acme:doc-1", "globex:doc-9"}))
    results = service.local_search(corpus_payloads, scope, "계약 예산", 50)
    assert all(not r["doc_id"].startswith("globex:") for r in results)


# 7. 필터에 tenant 조건이 반드시 포함됩니다
def test_filter_contains_tenant():
    f = build_filter(ACME)
    keys = [c.key for c in f.must]
    assert "tenant_id" in keys


# 8. 필터가 실제 검색 호출까지 전달됩니다 (구현 검사)
def test_filter_reaches_query(monkeypatch):
    captured = {}

    def fake_query(scope, query_filter, query, top_k):
        captured.update(scope=scope, filter=query_filter, top_k=top_k)
        return []

    monkeypatch.setattr(service, "_qdrant_query", fake_query)
    service.search(ACME, "계약")
    assert captured["scope"].tenant_id == "acme"
    assert captured["filter"] is not None


# 9~12. 메타 테스트: 일부러 새는 구현을 넣어 테스트가 잡는지 확인
@pytest.mark.parametrize("broken", ["drop_filter", "drop_prefetch_filter",
                                    "wrong_tenant", "no_payload_check"])
def test_meta_leak_is_caught(corpus_payloads, broken):
    assert isolation_violations(corpus_payloads, ACME) == []      # 정상 구현은 깨끗합니다
    with leaky(broken):
        violations = isolation_violations(corpus_payloads, ACME)
    assert violations, f"격리 검사가 {broken} 누출을 놓쳤습니다"


# 추가: allowlist를 빠뜨린 구현도 잡힙니다
def test_meta_allowlist_leak_is_caught(corpus_payloads):
    scope = TenantScope("acme", frozenset({"acme:doc-1"}))
    assert isolation_violations(corpus_payloads, scope, "계약 환불") == []
    with leaky("drop_allowlist"):
        assert isolation_violations(corpus_payloads, scope, "계약 환불")
