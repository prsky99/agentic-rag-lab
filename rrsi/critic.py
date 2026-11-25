"""평가 전 누출 스크리닝 (8.6절).

평가를 실행하기 전에 diff를 봅니다. 여기서 거르는 이유는 비용이 아니라 원칙입니다.
한 번 평가에 들어간 누출 후보는 점수가 좋게 나오고, 그 점수가 승격을 만듭니다.
나중에 잡으면 "무엇을 믿을 수 있었는지"를 다시 계산해야 합니다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Screening:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def changed_files(diff: str) -> list[str]:
    return re.findall(r"^\+\+\+ b/(.+)$", diff, flags=re.M)


def screen(diff: str, patterns: list[tuple[str, str]],
           harness_prefix: str = "domains/rag/harness/") -> Screening:
    reasons: list[str] = []

    # 1. 편집면 밖을 건드렸는가 — 구조적으로 가장 먼저 봅니다
    outside = [f for f in changed_files(diff) if not f.startswith(harness_prefix)]
    if outside:
        reasons.append("out-of-harness edit: " + ", ".join(outside))

    added = "\n".join(line[1:] for line in diff.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))

    # 2. 식별자·정답·평가 산출물 누출
    for pattern, why in patterns:
        m = re.search(pattern, added, flags=re.I | re.M)
        if m:
            reasons.append(f"{why}: {m.group(0)[:60]}")

    # 3. 아무 동작도 하지 않는 장치(죽은 분기) — 로그만 남기는 조건
    if re.search(r"if\s+.*:\s*\n\s*(log|print|logger)\.", added):
        reasons.append("dead branch: conditional only logs")

    return Screening(not reasons, reasons)
