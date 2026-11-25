"""orchestrator: 하네스를 HTTP로 감싼 서비스 (12.1절).

판단은 전부 `harness.pipeline`에 있습니다. 이 파일에는 인증·직렬화·에러 매핑만
둡니다. 서비스가 두꺼워지면 진화가 하네스 밖을 건드려야 하고, 그러면 critic의
편집 범위 검사가 무의미해집니다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "domains" / "rag"))
sys.path.insert(0, str(ROOT))

from harness.pipeline import RAGPipeline  # noqa: E402
from services.gateway.tier import request_policy  # noqa: E402

CONFIG = os.environ.get("HARNESS_CONFIG", str(ROOT / "domains/rag/harness/config.yaml"))
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "dev")

app = FastAPI(title="rag-orchestrator")
pipe: RAGPipeline | None = None


class AnswerRequest(BaseModel):
    question: str
    tenant_id: str            # 게이트웨이가 JWT 클레임에서 확정한 값
    tier: str = "free"
    deadline_ms: int | None = None   # None이면 설정의 기본값, 0 이하면 이미 만료(즉시 기권)


def _pipeline() -> RAGPipeline:
    global pipe
    if pipe is None:
        pipe = RAGPipeline.from_config(CONFIG)
    return pipe


def tenant_from_internal_token(x_internal_token: str = Header("")) -> str:
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(401, "invalid internal token")
    return "internal"


def _record_metrics(res: dict, tier: str) -> None:
    """도메인 지표(obs/metrics.py: 요청 수·지연·토큰·게이트 판정)를 기록합니다 (10.3절).

    관측은 부가 기능이므로 obs 패키지나 OTel이 없거나 기록 중 오류가 나도 응답은 그대로 나갑니다.
    """
    try:
        from obs.metrics import record
        record(res, "/v1/answer", 200, tier)
    except Exception:                               # noqa: BLE001
        pass


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "harness": _pipeline().policy_version}


@app.get("/ready")
def ready() -> dict:
    _pipeline()
    return {"ok": True}


@app.post("/v1/answer")
async def answer(req: AnswerRequest, _: str = Depends(tenant_from_internal_token)):
    pipe = _pipeline()
    # 등급 정책은 요청마다 계산해 넘깁니다. 공유 파이프라인 설정은 바꾸지 않습니다.
    policy = request_policy(pipe.cfg, req.tier)
    try:
        res = await pipe.answer(req.question, tenant=req.tenant_id, tenant_tier=req.tier,
                                deadline_ms=req.deadline_ms, policy=policy)
    except TimeoutError as exc:
        # 파이프라인은 시간 초과를 기권(deadline_exceeded)으로 끝냅니다. 그 밖의 경로에서
        # 시간 초과가 올라오면 500이 아니라 504로 답합니다.
        return JSONResponse({"status": "abstained", "abstained": True, "answer": "",
                             "citations": [], "reason": "deadline_exceeded",
                             "detail": str(exc)[:200]}, status_code=504)
    _record_metrics(res, req.tier)                  # 응답 직전 한 곳에서만 기록합니다
    return {"answer": res["answer"], "citations": res["citations"],
            "abstained": res["abstained"], "status": res["status"],
            "reason": res["reason"], "decisions": res["decisions"],
            "usage": res["usage"], "latency_ms": res["latency_ms"],
            "policy_version": res["policy_version"], "policy": res["policy"],
            "flags": res["flags"]}
