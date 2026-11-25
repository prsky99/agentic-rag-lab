#!/usr/bin/env python3
"""RAGAS 4지표 (9.3절).

RAGAS와의 유일한 접점입니다. 버전이 바뀌면 이 파일만 고치면 됩니다.
라가스가 설치되지 않은 환경에서는 지표를 계산하지 않고 그 사실을 출력합니다.

심판 LLM은 OpenAI 호환 엔드포인트면 무엇이든 쓸 수 있습니다(vLLM·SGLang·사내 게이트웨이 포함).
엔드포인트는 환경변수로 고릅니다.

    RAGAS_JUDGE_BASE_URL  (없으면 OPENAI_BASE_URL, 둘 다 없으면 api.openai.com)
    RAGAS_JUDGE_API_KEY   (없으면 OPENAI_API_KEY)

이 심판은 보상 계산용 정답 판정(rrsi.json의 judge_model, 결정 모델)과 다른 모델입니다.

    python eval/ragas_metrics.py --report /tmp/report.json \
        --judge-model gpt-6-luna --embed-model text-embedding-3-small
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "domains" / "rag"))
from outcomes import FAILURES, REFUSALS, outcome  # noqa: E402  (회귀 평가와 같은 분류, R01)


def judge_client():
    """RAGAS 심판용 비동기 클라이언트. base_url을 명시해 OpenAI 호환 서버를 가리킵니다."""
    from openai import AsyncOpenAI

    base_url = (os.environ.get("RAGAS_JUDGE_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL") or None)
    api_key = (os.environ.get("RAGAS_JUDGE_API_KEY")
               or os.environ.get("OPENAI_API_KEY") or "EMPTY")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


async def build(judge_model: str, embed_model=None, client=None):
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness)

    llm = llm_factory(judge_model, client=client or judge_client())
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
    if embed_model is not None:         # answer_relevancy는 임베딩이 있어야 계산됩니다
        metrics["answer_relevancy"] = AnswerRelevancy(llm=llm, embeddings=embed_model)
    return metrics


def build_embeddings(model: str, client=None):
    """임베딩 객체. 실패하면 None을 돌려주고 answer_relevancy를 건너뜁니다."""
    try:
        from ragas.embeddings import embedding_factory
        return embedding_factory("openai", model=model, client=client or judge_client())
    except Exception as exc:                                    # noqa: BLE001
        print(f"[ragas] 임베딩 생성 실패({exc}) — answer_relevancy는 건너뜁니다")
        return None


async def score_item(metrics, q, answer, contexts, reference) -> dict:
    out = {}
    f = await metrics["faithfulness"].ascore(
        user_input=q, response=answer, retrieved_contexts=contexts)
    out["faithfulness"] = f.value
    if "answer_relevancy" in metrics:
        r = await metrics["answer_relevancy"].ascore(user_input=q, response=answer)
        out["answer_relevancy"] = r.value
    else:
        out["answer_relevancy"] = None
    cp = await metrics["context_precision"].ascore(
        user_input=q, retrieved_contexts=contexts, reference=reference)
    out["context_precision"] = cp.value
    cr = await metrics["context_recall"].ascore(
        user_input=q, retrieved_contexts=contexts, reference=reference)
    out["context_recall"] = cr.value
    return out


async def eval_pipeline(run: dict, item: dict, metrics, contexts: list[str]) -> dict:
    kind = outcome(run)
    if not item["answerable"]:
        # 운영 장애·예산 초과로 끝난 기권은 기권 판단을 한 것이 아니므로 맞은 기권이 아닙니다(R01).
        return {"abstain_correct": 1.0 if kind in REFUSALS else 0.0,
                "faithfulness": None, "answer_relevancy": None,
                "context_precision": None, "context_recall": None}
    if run["abstained"] or kind in FAILURES:
        return {"abstain_correct": 0.0, "faithfulness": None, "answer_relevancy": None,
                "context_precision": None, "context_recall": None}
    scores = await score_item(metrics, item["question"], run["answer"], contexts,
                              item["reference"])
    return {"abstain_correct": 1.0, **scores}   # 답할 문항에 답했으므로 기권 판단은 맞습니다


def by_type(rows: list[dict]) -> dict:
    out = {}
    for t in {r["type"] for r in rows}:
        rs = [r for r in rows if r["type"] == t]

        def mean(key):                      # None(기권·미계산)은 분모에서 뺍니다
            vals = [r[key] for r in rs if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        out[t] = {"n": len(rs), "correct": mean("correct"),
                  "faithfulness": mean("faithfulness"),
                  "context_recall": mean("context_recall"),
                  "abstain": mean("abstain_correct")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="/tmp/report.json")
    ap.add_argument("--judge-model", default="gpt-6-luna")
    ap.add_argument("--embed-model", default="",
                    help="비우면 answer_relevancy 계산을 건너뜁니다")
    ap.add_argument("--out", default="eval/reports/ragas.json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # 윈도 cp949 콘솔에서 —·→ 출력 때문에 중단되지 않도록
        sys.stdout.reconfigure(errors="replace")

    try:
        import ragas  # noqa: F401
    except ImportError:
        print("[ragas] 설치되지 않음 — requirements-eval.txt를 설치해 주세요")
        return 0

    import asyncio
    if not Path(a.report).exists():
        print(f"[ragas] 리포트 없음: {a.report} — 먼저 "
              f"`python eval/run_subset.py --out {a.report}`로 만들어 주세요")
        return 1
    if not any(os.environ.get(k) for k in ("RAGAS_JUDGE_BASE_URL", "OPENAI_BASE_URL",
                                           "RAGAS_JUDGE_API_KEY", "OPENAI_API_KEY")):
        print("[ragas] 심판 LLM이 설정되지 않음 — 지표 계산을 건너뜁니다.\n"
              "        OpenAI: export OPENAI_API_KEY=sk-...  (또는 RAGAS_JUDGE_API_KEY)\n"
              "        사내 vLLM 등 OpenAI 호환 서버: export RAGAS_JUDGE_BASE_URL=http://<호스트>:8000/v1")
        return 2
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    goldenset = Path(__file__).resolve().parents[1] / "domains/rag/data/goldenset_v1.jsonl"
    items = {json.loads(l)["id"]: json.loads(l) for l in
             goldenset.read_text(encoding="utf-8").splitlines()
             if l.strip()}

    async def run() -> None:
        client = judge_client()
        emb = build_embeddings(a.embed_model, client) if a.embed_model else None
        metrics = await build(a.judge_model, emb, client)
        rows = []
        for r in report.get("items", []):
            item = items.get(r["id"])
            if not item:
                continue
            # 리랭킹 후 순서의 문맥 텍스트(run_subset.py가 contexts로 남깁니다)
            sc = await eval_pipeline(r, item, metrics, list(r.get("contexts") or []))
            rows.append({"id": r["id"], "type": item["type"], "correct": r.get("correct"),
                         **sc})
        out = {"n": len(rows), "judge_model": a.judge_model,
               "judge_base_url": str(client.base_url),
               "embed_model": a.embed_model or None,
               "by_type": by_type(rows), "items": rows}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ragas] {len(rows)}문항 → {a.out}")

    try:
        asyncio.run(run())
    except Exception as exc:             # noqa: BLE001  키·주소·모델 이름이 틀리면 첫 심판 호출에서 실패합니다
        # openai 예외는 ragas가 쓰는 instructor의 재시도 예외로 감싸져 올라옵니다
        if type(exc).__module__.split(".")[0] not in ("openai", "instructor", "httpx", "httpcore"):
            raise
        print(f"[ragas] 심판 호출 실패({type(exc).__name__}: {str(exc)[:200]})\n"
              "        API 키(RAGAS_JUDGE_API_KEY/OPENAI_API_KEY), 주소(RAGAS_JUDGE_BASE_URL), "
              "--judge-model 이름을 확인해 주세요")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
