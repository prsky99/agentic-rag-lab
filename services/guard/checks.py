"""출력 가드 (11.4절).

세 가지를 봅니다: 프롬프트 누출, 비밀 패턴, 인용 지지. 앞의 둘은 차단하고,
마지막은 재생성을 요청합니다.
"""
from __future__ import annotations

import re

CANARY = "ZQ7-4f19-KLM"      # 시스템 프롬프트에만 존재
SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{20,}|\d{6}-?\d{7}|(?:\d[ -]?){13,16}")
SPAN_OK = 0.6


def citations_supported(answer: str, citations: list[dict], chunks: dict[str, str]) -> bool:
    if not citations:
        return False
    for c in citations:
        text = chunks.get(c.get("chunk_id", ""), "")
        span = c.get("span", "")
        if not span or not text or span not in text:
            return False
    return True


def output_guard(answer: str, citations: list[dict], chunks: dict[str, str]) -> tuple[str, str | None]:
    if CANARY in (answer or ""):
        return "block", "prompt_leak"
    if SECRET_RE.search(answer or ""):          # API 키·주민번호·카드번호 패턴
        return "block", "secret_pattern"
    if not citations_supported(answer, citations, chunks):
        return "regenerate", "unsupported_citation"
    return "ok", None
