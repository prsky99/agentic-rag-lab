"""파이프라인 출력 가드 (11.4절).

독립 가드 서비스(``services/guard/checks.py``)와 같은 차단 규칙을 하네스 안에서
적용합니다. 파이프라인은 최종 응답을 내보내기 직전(최초 생성·재생성·무검색 답변 등
모든 ``answered`` 경로)에 이 함수를 반드시 거칩니다.

차단 대상은 두 가지입니다.

* 프롬프트 누출: 시스템 프롬프트에만 있는 카나리 문자열
* 비밀 패턴: API 키·주민등록번호·카드번호 모양의 문자열

답변 본문과 함께 응답에 실리는 인용 구간(span)도 검사합니다. 인용 지지 여부는
G3(충실성 게이트)가 판정하므로 여기서는 다루지 않습니다.
"""
from __future__ import annotations

import re

CANARY = "ZQ7-4f19-KLM"      # 시스템 프롬프트에만 존재(services/guard/checks.py와 같은 값)
SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{20,}|\d{6}-?\d{7}|(?:\d[ -]?){13,16}")


def output_block_reason(answer: str, spans: list[str] | None = None) -> str | None:
    """차단해야 하면 사유(prompt_leak|secret_pattern)를, 아니면 None을 돌려줍니다."""
    texts = [answer or ""] + [s or "" for s in (spans or [])]
    if any(CANARY in t for t in texts):
        return "prompt_leak"
    if any(SECRET_RE.search(t) for t in texts):
        return "secret_pattern"
    return None
