"""임베딩 모델 설정의 단일 출처 (3.2절, 3.4절).

색인기(``eval/index.py``, ``services/retriever/index.py``)와 검색기
(``services/retriever/service.py``)는 모델 이름과 차원을 이 파일에서만 읽습니다. 색인할 때와
검색할 때 모델이 다르면 벡터 공간이 달라 검색 결과가 무의미해지고, 차원이 다르면 Qdrant가
요청을 거절합니다.

환경 변수(없으면 기본값)

* ``EMBED_MODEL``: dense 모델 이름. 기본값은
  ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``입니다
  (384차원, 약 0.22GB, Apache-2.0, 한국어를 포함한 50여 개 언어).
  예전 이름 ``DENSE_MODEL``도 읽습니다(``EMBED_MODEL``이 우선).
* ``EMBED_DIM``: dense 벡터 차원. 비우면 설치된 fastembed의 지원 목록에서 찾습니다.
  값을 주면 모델의 실제 차원과 같은지 적재·검색 때 확인합니다.
* ``SPARSE_MODEL``: sparse 모델 이름. 기본값 ``Qdrant/bm25``.
* ``EMBED_LOCAL_FILES_ONLY``: ``1``이면 모델을 내려받지 않고 캐시에서만 읽습니다
  (모델을 이미지에 넣어 두었거나 외부 통신을 막은 환경).
* ``FASTEMBED_CACHE_PATH``: 모델 캐시 폴더. fastembed가 직접 읽습니다. 비우면 임시 폴더
  (``/tmp/fastembed_cache``)에 내려받으므로, 루트 파일 시스템이 읽기 전용인 컨테이너에서는
  쓰기 가능한 볼륨 경로로 지정해야 합니다.

기본 모델을 고른 근거: requirements에 고정된 ``fastembed==0.4.2``의 ``TextEmbedding``은
``BAAI/bge-m3``를 지원하지 않습니다. 지원 목록의 다국어 dense 모델은 세 가지이고
(MiniLM-L12 384차원 0.22GB, mpnet-base 768차원 1.0GB, multilingual-e5-large 1024차원 2.24GB),
실습 이미지와 노트북에서 부담 없이 받을 수 있는 가장 작은 모델을 기본값으로 둡니다.
품질이 더 필요하면 ``EMBED_MODEL=intfloat/multilingual-e5-large``로 바꾸고 새 컬렉션에
다시 색인합니다. e5 계열은 질의에 ``query: ``, 문서에 ``passage: `` 접두사를 붙여야 하므로
``load_models``가 그 접두사를 붙이는 래퍼를 돌려줍니다.
"""
from __future__ import annotations

import os

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBED_DIM = 384
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


def embed_model() -> str:
    return (os.environ.get("EMBED_MODEL") or os.environ.get("DENSE_MODEL")
            or DEFAULT_EMBED_MODEL)


def sparse_model() -> str:
    return os.environ.get("SPARSE_MODEL") or DEFAULT_SPARSE_MODEL


def supported_dense() -> dict[str, int]:
    """설치된 fastembed가 지원하는 dense 모델 → 차원. fastembed가 없으면 ImportError입니다."""
    from fastembed import TextEmbedding
    return {m["model"]: int(m["dim"]) for m in TextEmbedding.list_supported_models()}


def supported_sparse() -> set[str]:
    from fastembed import SparseTextEmbedding
    return {m["model"] for m in SparseTextEmbedding.list_supported_models()}


def embed_dim() -> int:
    """dense 벡터 차원. ``EMBED_DIM`` → fastembed 지원 목록 → 기본 모델의 차원 순으로 정합니다."""
    raw = os.environ.get("EMBED_DIM", "").strip()
    if raw:
        dim = int(raw)
        if dim <= 0:
            raise ValueError(f"EMBED_DIM은 양의 정수여야 합니다: {raw}")
        return dim
    name = embed_model()
    if name == DEFAULT_EMBED_MODEL:
        return DEFAULT_EMBED_DIM
    try:
        known = supported_dense()
    except ImportError as exc:
        raise ValueError(f"{name}의 차원을 알 수 없습니다. EMBED_DIM을 지정해 주세요") from exc
    if name not in known:
        raise ValueError(f"{name}의 차원을 알 수 없습니다. EMBED_DIM을 지정해 주세요")
    return known[name]


def check_supported() -> None:
    """모델 이름·차원이 설치된 fastembed와 맞는지 확인합니다. 틀리면 ValueError입니다."""
    name, dense, sparse = embed_model(), supported_dense(), supported_sparse()
    if name not in dense:
        raise ValueError(
            f"dense 모델 {name}은(는) 설치된 fastembed가 지원하지 않습니다. "
            f"EMBED_MODEL을 지원 목록(TextEmbedding.list_supported_models()) 안에서 골라 주세요")
    if embed_dim() != dense[name]:
        raise ValueError(f"EMBED_DIM={embed_dim()}이 {name}의 차원 {dense[name]}과 다릅니다")
    if sparse_model() not in sparse:
        raise ValueError(f"sparse 모델 {sparse_model()}은(는) 설치된 fastembed가 지원하지 않습니다")


def prefixes(name: str) -> tuple[str, str]:
    """(질의 접두사, 문서 접두사). e5 계열만 접두사를 씁니다."""
    return ("query: ", "passage: ") if "e5" in name.lower() else ("", "")


class Prefixed:
    """질의·문서에 접두사를 붙여 임베딩 모델을 부릅니다(fastembed와 같은 메서드 모양)."""

    def __init__(self, model, query_prefix: str, passage_prefix: str) -> None:
        self.model, self.query_prefix, self.passage_prefix = model, query_prefix, passage_prefix

    def embed(self, texts, **kw):
        return self.model.embed([self.passage_prefix + t for t in texts], **kw)

    def query_embed(self, query, **kw):
        queries = [query] if isinstance(query, str) else list(query)
        return self.model.embed([self.query_prefix + q for q in queries], **kw)


def load_models():
    """(dense, sparse) 임베딩 모델. 색인기와 검색기가 같은 이 함수를 씁니다.

    모델 파일이 캐시에 없으면 처음 한 번 내려받습니다(``EMBED_LOCAL_FILES_ONLY=1``이면
    내려받지 않고 실패합니다).
    """
    from fastembed import SparseTextEmbedding, TextEmbedding
    check_supported()
    kw = {"local_files_only": True} \
        if os.environ.get("EMBED_LOCAL_FILES_ONLY", "").lower() in ("1", "true", "yes") else {}
    name = embed_model()
    dense = TextEmbedding(name, **kw)
    q, p = prefixes(name)
    if q or p:
        dense = Prefixed(dense, q, p)
    return dense, SparseTextEmbedding(sparse_model(), **kw)
