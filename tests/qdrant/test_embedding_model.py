"""임베딩 모델 설정 시험(2차 검토 D01).

기본 dense 모델이 requirements에 고정된 fastembed의 지원 목록에 있어야 색인 CLI가 돕니다.
세 단계로 확인합니다.

1. 설정 모듈만 보는 시험: 패키지 없이 실행합니다.
2. 지원 목록 시험: fastembed가 설치돼 있을 때만 실행합니다(모델을 내려받지 않습니다).
3. 실제 모델 시험: fastembed·qdrant-client가 있고 모델 파일이 캐시에 있을 때만 실행합니다.
   캐시에 없으면 건너뜁니다. ``RUN_EMBED_DOWNLOAD=1``이면 내려받아(약 0.22GB) 실행합니다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from services.retriever import embedding

ROOT = Path(__file__).resolve().parents[2]
REFUND_Q = "환불은 결제일로부터 며칠 이내에 요청할 수 있나요?"


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    for name in ("EMBED_MODEL", "DENSE_MODEL", "EMBED_DIM", "SPARSE_MODEL"):
        monkeypatch.delenv(name, raising=False)


# --- 1. 설정 모듈 -----------------------------------------------------------
def test_d01_single_source_defaults():
    assert embedding.embed_model() == embedding.DEFAULT_EMBED_MODEL
    assert embedding.embed_dim() == embedding.DEFAULT_EMBED_DIM == 384
    assert embedding.sparse_model() == "Qdrant/bm25"
    assert embedding.DEFAULT_EMBED_MODEL != "BAAI/bge-m3"


def test_d01_env_overrides_and_legacy_name(monkeypatch):
    monkeypatch.setenv("DENSE_MODEL", "legacy/model")
    assert embedding.embed_model() == "legacy/model"
    monkeypatch.setenv("EMBED_MODEL", "new/model")
    assert embedding.embed_model() == "new/model"          # EMBED_MODEL이 우선
    monkeypatch.setenv("EMBED_DIM", "768")
    assert embedding.embed_dim() == 768
    monkeypatch.setenv("EMBED_DIM", "0")
    with pytest.raises(ValueError):
        embedding.embed_dim()


def test_d01_indexer_and_searcher_read_the_same_module():
    """색인기와 검색기에 모델 이름이 따로 적혀 있으면 안 됩니다."""
    for rel in ("eval/index.py", "services/retriever/index.py", "services/retriever/service.py",
                "services/retriever/search.py", "services/retriever/store.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "bge-m3" not in src and "TextEmbedding(" not in src, rel
        assert "embedding" in src, rel


def test_d01_e5_models_get_prefixes():
    assert embedding.prefixes("intfloat/multilingual-e5-large") == ("query: ", "passage: ")
    assert embedding.prefixes(embedding.DEFAULT_EMBED_MODEL) == ("", "")

    class Echo:
        def embed(self, texts, **kw):
            return list(texts)

    m = embedding.Prefixed(Echo(), "query: ", "passage: ")
    assert m.embed(["가", "나"]) == ["passage: 가", "passage: 나"]
    assert m.query_embed("질문") == ["query: 질문"]


# --- 2. 설치된 fastembed의 지원 목록 ------------------------------------------
def test_d01_default_models_are_supported_by_installed_fastembed():
    pytest.importorskip("fastembed")
    dense = embedding.supported_dense()
    assert embedding.DEFAULT_EMBED_MODEL in dense
    assert dense[embedding.DEFAULT_EMBED_MODEL] == embedding.DEFAULT_EMBED_DIM
    assert embedding.DEFAULT_SPARSE_MODEL in embedding.supported_sparse()
    embedding.check_supported()                          # 예외 없이 통과


def test_d01_unsupported_model_is_reported_before_download(monkeypatch):
    pytest.importorskip("fastembed")
    monkeypatch.setenv("EMBED_MODEL", "BAAI/bge-m3")
    if "BAAI/bge-m3" in embedding.supported_dense():
        pytest.skip("설치된 fastembed가 bge-m3를 지원합니다")
    with pytest.raises(ValueError, match="지원하지 않습니다"):
        embedding.check_supported()


def test_d01_dim_mismatch_is_rejected(monkeypatch):
    pytest.importorskip("fastembed")
    monkeypatch.setenv("EMBED_DIM", "1024")
    with pytest.raises(ValueError, match="차원"):
        embedding.check_supported()


def test_d01_index_cli_exits_2_for_unsupported_model(tmp_path):
    pytest.importorskip("fastembed")
    pytest.importorskip("qdrant_client")
    if "BAAI/bge-m3" in embedding.supported_dense():
        pytest.skip("설치된 fastembed가 bge-m3를 지원합니다")
    proc = subprocess.run(
        [sys.executable, "-m", "eval.index", "--qdrant-url", ":memory:", "--out", str(tmp_path)],
        cwd=str(ROOT), env={**os.environ, "EMBED_MODEL": "BAAI/bge-m3", "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 2 and "지원하지 않습니다" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_d01_collection_dim_comes_from_settings_and_mismatch_fails(monkeypatch):
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient
    from services.retriever.index import dense_size, ensure_collection, upsert_chunks
    client = QdrantClient(location=":memory:")
    ensure_collection(client, "c1")                      # dim 인자 없이 설정의 차원
    assert dense_size(client, "c1") == embedding.DEFAULT_EMBED_DIM
    monkeypatch.setenv("EMBED_DIM", "1024")
    with pytest.raises(ValueError, match="다시 색인"):
        ensure_collection(client, "c1")                  # 모델을 바꿨는데 같은 컬렉션

    class Dense:
        def embed(self, texts, **kw):
            return [[0.1] * 8 for _ in texts]

    class Sparse:
        def embed(self, texts, **kw):
            from types import SimpleNamespace
            return [SimpleNamespace(indices=[1], values=[1.0]) for _ in texts]

    chunk = {"chunk_id": "acme/a#0", "doc_id": "acme/a", "tenant": "acme", "text": "본문"}
    with pytest.raises(ValueError, match="설정 차원"):
        upsert_chunks(client, "c2", [chunk], Dense(), Sparse(), dim=1024)
    assert not client.collection_exists("c2")


# --- 3. 실제 모델: 33청크 색인 → 예시 질문 검색 --------------------------------
@pytest.fixture(scope="module")
def real_models():
    pytest.importorskip("fastembed")
    pytest.importorskip("qdrant_client")
    saved = {k: os.environ.pop(k, None) for k in ("EMBED_MODEL", "DENSE_MODEL", "EMBED_DIM",
                                                  "SPARSE_MODEL", "EMBED_LOCAL_FILES_ONLY")}
    try:
        if os.environ.get("RUN_EMBED_DOWNLOAD") != "1":
            os.environ["EMBED_LOCAL_FILES_ONLY"] = "1"   # 캐시에 없으면 내려받지 않고 건너뜁니다
        try:
            return embedding.load_models()
        except Exception as exc:                         # noqa: BLE001
            pytest.skip(f"임베딩 모델이 캐시에 없습니다({type(exc).__name__}). "
                        "RUN_EMBED_DOWNLOAD=1로 내려받아 실행할 수 있습니다")
    finally:
        os.environ.pop("EMBED_LOCAL_FILES_ONLY", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_d01_real_model_indexes_33_chunks_and_finds_refund_policy(real_models):
    from qdrant_client import QdrantClient

    from eval.index import load_to_qdrant
    from harness.retrieve import chunk_doc, load_corpus
    from services.retriever.index import dense_size
    from services.retriever.scope import TenantScope
    from services.retriever.search import hybrid_search

    dense, sparse = real_models
    passages = [p.as_dict() for doc in load_corpus(ROOT / "domains/rag/data/corpus")
                if doc["tenant"] == "acme" for p in chunk_doc(doc)]
    assert len(passages) == 33
    client = QdrantClient(location=":memory:")
    n = load_to_qdrant(client, passages, "demo", "idx-real", "docs_current", dense, sparse,
                       dim=embedding.embed_dim())
    assert n == 33 and client.count("idx-real").count == 33
    assert dense_size(client, "docs_current") == embedding.embed_dim() == 384

    hits = hybrid_search(client, "docs_current", TenantScope("demo"), REFUND_Q, dense, sparse,
                         top_k=5)
    ids = [h["chunk_id"] for h in hits]
    assert "acme/refund-policy#1" in ids[:3], ids
    # dense만으로도 상위에 와야 합니다(색인과 질의가 같은 모델·차원을 쓴다는 확인).
    qv = next(iter(dense.query_embed(REFUND_Q))).tolist()
    assert len(qv) == 384
    only_dense = client.query_points("docs_current", query=qv, using="dense", limit=5,
                                     with_payload=True).points
    assert "acme/refund-policy#1" in [p.payload["chunk_id"] for p in only_dense][:5]
