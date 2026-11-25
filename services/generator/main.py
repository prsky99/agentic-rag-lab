"""생성 서비스 (6.4절): /generate.

프롬프트는 하네스가 만듭니다. 이 서비스는 모델 호출과 토큰 회계만 합니다.

계약(하네스 ``LLMClient(provider="generator")``와 같습니다)
    요청: ``POST /generate`` ``{system, user, temperature, max_tokens}``
    헤더: ``x-deadline-ms``(선택) — 상위 요청의 남은 시간(ms)입니다. 이 값이 있으면 제공자
        호출의 전체 제한 시간을 그 값으로 줄입니다(기본 45초보다 길어지지는 않습니다).
        0 이하이면 제공자를 부르지 않고 ``error="deadline_exceeded"``를 돌려줍니다.
    응답: ``{answer, usage: {input_tokens, output_tokens, source}, error}``
    ``source``는 제공자가 돌려준 실제 사용량이면 ``provider``(실측)이고, 제공자가 사용량을
    주지 않았거나 호출이 실패했으면 ``unknown``(미측정)입니다. 미측정일 때 토큰 값은 글자 수
    추정치(실패하면 0)이므로 받는 쪽은 실측으로 세지 않습니다.

이 서비스 자신은 항상 외부 모델 제공자(OpenAI 호환 ``/chat/completions``)를 부릅니다.
주소는 ``OPENAI_BASE_URL``, 키는 ``GENERATOR_API_KEY``(없으면 ``OPENAI_API_KEY``)입니다.
``GENERATOR_URL``은 이 서비스를 가리키므로 여기서는 쓰지 않습니다(자기 자신 호출 방지).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Header
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "domains" / "rag"))

from harness.llm import LLMClient, LLMUnavailable, call_scope  # noqa: E402

app = FastAPI(title="rag-generator")


class GenerateRequest(BaseModel):
    system: str
    user: str
    temperature: float = 0.0
    max_tokens: int = 700


def provider_client() -> LLMClient:
    return LLMClient(base_url=os.environ.get("OPENAI_BASE_URL", ""),
                     api_key=(os.environ.get("GENERATOR_API_KEY")
                              or os.environ.get("OPENAI_API_KEY", "")),
                     provider="openai")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "mode": provider_client().mode}


def remaining_s(x_deadline_ms: str | None) -> float | None:
    """``x-deadline-ms`` 헤더 → 남은 시간(초). 없거나 숫자가 아니면 None(기본 타임아웃)입니다."""
    try:
        return None if x_deadline_ms is None else float(x_deadline_ms) / 1000.0
    except ValueError:
        return None


NO_USAGE = {"input_tokens": 0, "output_tokens": 0, "source": "unknown"}


@app.post("/generate")
def generate(req: GenerateRequest, x_deadline_ms: str | None = Header(None)) -> dict:
    left = remaining_s(x_deadline_ms)
    if left is not None and left <= 0:
        return {"answer": "", "error": "deadline_exceeded", "usage": dict(NO_USAGE)}
    client = provider_client()
    try:
        # 상위의 남은 시간을 제공자 호출의 전체 제한 시간으로 씁니다(RT04).
        with call_scope(left):
            text, usage = client.complete_with_usage(req.system, req.user, req.temperature,
                                                     req.max_tokens)
    except LLMUnavailable as exc:
        return {"answer": "", "error": str(exc), "usage": dict(NO_USAGE)}
    return {"answer": text, "usage": usage, "error": None}
