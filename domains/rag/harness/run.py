"""실행 진입점 (8.2절).

평가 하네스가 subprocess로 부릅니다.

    python -m harness.run --ids /tmp/ids.json --out runs/rag/jobs/r0/A/s0 --seed 0

출력 스키마는 편집 금지입니다(SKILL.md 금지 3, 8.10절). 채점기와 제안자가 이 키에 의존합니다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):                     # python harness/run.py 로 직접 실행
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness.pipeline import RAGPipeline     # type: ignore
else:
    from .pipeline import RAGPipeline


async def one(pipe: RAGPipeline, task: dict, out_dir: Path, sem: asyncio.Semaphore) -> None:
    async with sem:
        try:
            res = await pipe.answer(task["question"], tenant=task.get("tenant", "acme"),
                                    tenant_tier=task.get("tenant_tier", "free"))
        except Exception as exc:                   # noqa: BLE001
            res = {"answer": "", "abstained": True, "citations": [], "retrieved": [],
                   "decisions": {}, "usage": {}, "error": repr(exc)}
        res["task_id"] = task["id"]
        res["latency_ms"] = res.get("latency_ms", 0)
        res.setdefault("retrieved", [])
        (out_dir / f"{task['id']}.json").write_text(
            json.dumps(res, ensure_ascii=False), encoding="utf-8")


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="문항 JSON [{id,question,tenant}]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--index-snapshot", default=os.environ.get("RAG_INDEX_SNAPSHOT"))
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path(a.ids).read_text(encoding="utf-8"))
    pipe = RAGPipeline.from_config(a.config, seed=a.seed, index_snapshot=a.index_snapshot)
    sem = asyncio.Semaphore(max(1, a.concurrency))
    await asyncio.gather(*(one(pipe, t, out, sem) for t in tasks))
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except Exception:                              # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
