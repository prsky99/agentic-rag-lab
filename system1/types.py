"""게이트가 주고받는 값 타입 (4.4절).

검색 결과 문단과 인용을 같은 모양으로 고정해 두면, 게이트·평가 하네스·데모가
같은 객체를 씁니다. ``chunk_id``가 기본 키이고 ``doc_id``는 집계용입니다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    text: str
    doc_id: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    span: str          # 인용 구간을 식별하는 문자열(문단 내 오프셋 또는 인용문)
    text: str
