"""코어 공용 유틸(로깅·JSON). 도메인 지식은 도메인 쪽에만 둡니다."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def log(domain: str, msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[rrsi:{domain}] {stamp} {msg}", file=sys.stdout, flush=True)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
