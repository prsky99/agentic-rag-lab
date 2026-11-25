"""트레이스 요약기 (7.2절).

원본 트레이스는 깁니다. 제안자에게 넣을 수 있는 길이로 줄이되, 판단에 필요한
필드(경로·게이트 판정·토큰·지연)는 남깁니다.
"""
from __future__ import annotations

from pathlib import Path

from ._log import read_json


def digest_trial(path: Path) -> str:
    t = read_json(path, {}) or {}
    d = t.get("decisions") or {}
    g1 = d.get("g1") or {}
    g2 = d.get("g2") or {}
    g3 = d.get("g3") or {}
    usage = t.get("usage") or {}
    return (f"{t.get('task_id', path.stem)}: route={g1.get('choice')} "
            f"g2={g2.get('verdict')}({g2.get('p_yes')}) "
            f"g3={g3.get('ok')} abstain={t.get('abstained')} "
            f"tok={usage.get('total_tokens')} {t.get('latency_ms')}ms")


def digest_job(runs_dir: Path, task_ids: list[str], k: int = 1,
               limit: int = 12) -> list[str]:
    lines = []
    for tid in task_ids[:limit]:
        p = Path(runs_dir) / "jobs" / tid
        if p.is_file():
            lines.append(digest_trial(p))
    return lines
