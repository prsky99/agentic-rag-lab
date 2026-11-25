"""테스트 공용 픽스처.

테넌트 격리 테스트는 qdrant 없이 메모리 백엔드로 실행합니다. 필터 객체와 적용기가
운영과 같은 코드이므로, 여기서 통과하면 저장소 계층에서도 같은 규칙이 지켜집니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "domains" / "rag"))


@pytest.fixture
def corpus_payloads() -> list[dict]:
    return [
        {"chunk_id": "acme:doc-1#0", "doc_id": "acme:doc-1", "tenant_id": "acme",
         "title": "계약", "section": "해지", "text": "계약 해지는 30일 전에 통지한다."},
        {"chunk_id": "acme:doc-1#1", "doc_id": "acme:doc-1", "tenant_id": "acme",
         "title": "계약", "section": "갱신", "text": "계약 갱신은 만료 60일 전에 협의한다."},
        {"chunk_id": "acme:doc-2#0", "doc_id": "acme:doc-2", "tenant_id": "acme",
         "title": "환불", "section": "기한", "text": "환불은 14일 이내에 요청한다."},
        {"chunk_id": "globex:doc-9#0", "doc_id": "globex:doc-9", "tenant_id": "globex",
         "title": "청룡", "section": "예산",
         "text": "청룡-7731 프로젝트 계약 예산은 12억 원이다."},
    ]


@pytest.fixture
def pipeline():
    from harness.pipeline import RAGPipeline
    return RAGPipeline.from_config(ROOT / "domains/rag/harness/config.yaml")
