"""구조 인식 청킹 (3.3절).

마크다운 제목 계층으로 나누고, 상한을 넘으면 문단 단위로 자릅니다. 코드 펜스와
표는 절대 중간에서 자르지 않습니다. 표를 자르면 계산형 질문의 정답이 사라집니다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE = re.compile(r"^```")


@dataclass
class Chunk:
    text: str
    section: str
    kind: str          # prose | table | code


def split_structured(md: str, max_tokens: int = 700, overlap: int = 80,
                     tokens=lambda s: max(1, len(s) // 2)) -> list[Chunk]:
    """마크다운 제목 계층으로 나누고, 상한을 넘으면 문단 단위로 자릅니다.
    코드 펜스와 표는 절대 중간에서 자르지 않습니다."""
    chunks: list[Chunk] = []
    section, buf, kind = "", [], "prose"
    in_fence = False

    def flush():
        nonlocal buf, kind
        text = "\n".join(buf).strip()
        if not text:
            buf = []
            return
        if kind in ("code", "table") or tokens(text) <= max_tokens:
            chunks.append(Chunk(text, section, kind))
        else:
            # 문단 단위로 쪼개고, 상한을 넘을 때만 오버랩을 줍니다
            paras = [p for p in re.split(r"\n{2,}", text) if p.strip()]
            cur: list[str] = []
            for p in paras:
                if cur and tokens("\n\n".join(cur + [p])) > max_tokens:
                    body = "\n\n".join(cur)
                    chunks.append(Chunk(body, section, "prose"))
                    tail = body[-overlap * 2:] if overlap else ""
                    cur = [tail, p] if tail else [p]
                else:
                    cur.append(p)
            if cur:
                chunks.append(Chunk("\n\n".join(cur), section, "prose"))
        buf, kind = [], "prose"

    for line in md.splitlines():
        if FENCE.match(line):
            if not in_fence:
                # 여는 펜스: 앞의 산문·표를 먼저 내보내고 코드 블록을 새로 시작합니다.
                flush()
                in_fence, kind = True, "code"
                buf.append(line)
            else:
                # 닫는 펜스: 펜스까지 포함한 코드 블록 전체를 한 청크로 내보냅니다.
                buf.append(line)
                flush()
                in_fence = False
            continue
        if in_fence:
            buf.append(line)
            continue
        m = HEADING.match(line)
        if m and not in_fence:
            flush()
            section = m.group(2).strip()
            buf.append(line)
            continue
        if not in_fence and line.strip().startswith("|"):
            if kind != "table":
                flush()
                kind = "table"
            buf.append(line)
            continue
        if kind == "table" and not line.strip().startswith("|"):
            flush()
            kind = "prose"
        buf.append(line)
    flush()
    return chunks
