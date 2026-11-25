#!/usr/bin/env python3
"""색인 스냅샷 만들기 (3.4절).

    python -m eval.index --corpus domains/rag/data/corpus --tenant demo \
        --alias docs_current --snapshot idx-2026-11-20

로컬 스냅샷(JSON)은 언제나 남깁니다. 평가 하네스는 이 스냅샷을 읽어 같은 문단 구성을
재현합니다. ``QDRANT_URL``(또는 ``--qdrant-url``)이 있으면 같은 청크를 Qdrant에도 적재합니다.

* 컬렉션 이름은 스냅샷 이름(예: idx-2026-11-20), 읽기 별칭은 ``--alias``(docs_current)입니다.
  적재가 끝나면 별칭을 새 컬렉션으로 원자 교체합니다(3.3절).
* point ID는 chunk_id의 uuid5(결정적 UUID)이고, chunk_id는 payload에 남습니다.
* payload의 tenant_id는 로그인 테넌트(``--tenant``) 값입니다. 검색 필터가 JWT의 테넌트로
  거르기 때문입니다. 원문 문서의 테넌트는 source_tenant에 남깁니다.
* kind에서는 ``kubectl -n rag port-forward svc/qdrant 6333:6333``을 열어 둔 뒤
  ``QDRANT_URL=http://localhost:6333 make index``로 적재합니다. ``QDRANT_URL=:memory:``는
  프로세스 안 메모리 백엔드로 적재만 시험합니다.
* Qdrant 주소를 줬는데 qdrant-client·fastembed가 없거나 적재가 실패하면 종료 코드 2로
  끝납니다(조용히 로컬 스냅샷만 남기지 않습니다).
* 임베딩 모델과 차원은 검색 서비스와 같은 ``services/retriever/embedding.py``에서 읽습니다
  (``EMBED_MODEL``·``EMBED_DIM``·``SPARSE_MODEL``). 설치된 fastembed가 지원하지 않는 모델
  이름이면 내려받기 전에 종료 코드 2로 끝납니다. 처음 실행할 때 모델 파일(기본 모델 약
  0.22GB)을 ``FASTEMBED_CACHE_PATH``(없으면 임시 폴더)에 내려받습니다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "domains" / "rag"))

from harness.retrieve import chunk_doc, load_corpus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="domains/rag/data/corpus")
    ap.add_argument("--tenant", default="demo")
    ap.add_argument("--alias", default="docs_current")
    ap.add_argument("--snapshot", default=os.environ.get("RAG_INDEX_SNAPSHOT", "idx-local"))
    ap.add_argument("--out", default="domains/rag/data/index")
    ap.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", ""))
    a = ap.parse_args()

    alias = {"demo": {"acme"}, "other": {"globex"}}.get(a.tenant, {a.tenant})
    passages = []
    for doc in load_corpus(Path(a.corpus)):
        if doc["tenant"] not in alias:
            continue
        for p in chunk_doc(doc):
            passages.append(p.as_dict())

    out = Path(a.out) / f"{a.snapshot}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"alias": a.alias, "tenant": a.tenant,
                               "snapshot": a.snapshot, "passages": passages},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[index] {len(passages)}개 청크 → {out} (tenant {a.tenant})")

    if not a.qdrant_url:
        print("[index] QDRANT_URL 없음 → 로컬 스냅샷만 기록")
        return 0
    try:
        from qdrant_client import QdrantClient
        import fastembed  # noqa: F401
        from services.retriever.embedding import embed_dim, embed_model, load_models
    except ImportError as exc:
        print(f"[index] Qdrant 적재에 필요한 패키지가 없습니다({exc.name}). "
              "pip install -r requirements.txt 후 다시 실행해 주세요", file=sys.stderr)
        return 2
    try:
        dense, sparse = load_models()                 # 지원하지 않는 모델이면 여기서 끝납니다
        dim = embed_dim()
    except Exception as exc:                          # noqa: BLE001
        print(f"[index] 임베딩 모델을 준비하지 못했습니다: {exc}", file=sys.stderr)
        return 2
    client = (QdrantClient(location=":memory:") if a.qdrant_url == ":memory:"
              else QdrantClient(url=a.qdrant_url))
    try:
        n = load_to_qdrant(client, passages, a.tenant, a.snapshot, a.alias, dense, sparse,
                           dim=dim)
    except Exception as exc:                          # noqa: BLE001
        print(f"[index] Qdrant 적재 실패: {exc}", file=sys.stderr)
        return 2
    print(f"[index] Qdrant {a.qdrant_url}: {n}개 점 → 컬렉션 {a.snapshot}, 별칭 {a.alias} "
          f"(모델 {embed_model()}, {dim}차원)")
    return 0


def to_chunks(passages: list[dict], tenant: str) -> list[dict]:
    """스냅샷 문단 → 적재용 청크. tenant_id는 검색 필터가 쓰는 로그인 테넌트입니다."""
    return [{**p, "tenant_id": tenant, "source_tenant": p.get("tenant")} for p in passages]


def load_to_qdrant(client, passages: list[dict], tenant: str, collection: str,
                   alias: str, dense, sparse, dim: int | None = None) -> int:
    """컬렉션에 업서트하고 읽기 별칭을 그 컬렉션으로 옮깁니다. ``dim``은 기대 차원입니다."""
    from services.retriever.index import point_alias, upsert_chunks
    n = upsert_chunks(client, collection, to_chunks(passages, tenant), dense, sparse, dim=dim)
    if n:
        point_alias(client, alias, collection)
    return n


if __name__ == "__main__":
    raise SystemExit(main())
