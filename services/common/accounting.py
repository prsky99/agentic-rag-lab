"""요청 단위 토큰 회계 (2.5절, 10.6절)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Usage:
    stage: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class RequestCost:
    request_id: str
    usage: list[Usage] = field(default_factory=list)

    def add(self, u: Usage) -> None:
        self.usage.append(u)

    def total_tokens(self) -> int:
        return sum(u.input_tokens + u.output_tokens for u in self.usage)

    def by_stage(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for u in self.usage:
            out[u.stage] = out.get(u.stage, 0) + u.input_tokens + u.output_tokens
        return out
