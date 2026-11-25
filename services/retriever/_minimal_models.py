"""qdrant_client.models의 최소 대체 구현.

테스트와 로컬 실행에서 qdrant 패키지 없이 필터를 만들고 검사하기 위한 것입니다.
필드 이름과 모양은 qdrant와 같게 유지합니다(`.must`, `.key`, `.match.value`,
`.match.any`).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchValue:
    value: object


@dataclass
class MatchAny:
    any: list


@dataclass
class FieldCondition:
    key: str
    match: object


@dataclass
class Filter:
    must: list = field(default_factory=list)
    must_not: list = field(default_factory=list)
