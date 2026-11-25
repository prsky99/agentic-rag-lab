"""편집 이력 L_t, 정체 플래그, 미탐색 컴포넌트 (7.3절).

이력은 두 가지 용도로 씁니다.
1. 제안자에게 "이미 해 본 것"을 보여 주어 같은 편집의 반복을 막습니다.
2. 최근 라운드가 모두 탈락이면 정체(stagnation)로 보고 탐색을 넓히도록 지시합니다.
"""
from __future__ import annotations

import json
from pathlib import Path


class History:
    def __init__(self, path: Path, keep: int = 10) -> None:
        self.path = Path(path)
        self.keep = keep
        self.records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def append(self, record: dict) -> None:
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def stagnation(self, window: int = 3) -> bool:
        recent = self.records[-window:]
        if len(recent) < window:
            return False
        return not any(r.get("accepted") for r in recent)

    def touched(self, window: int = 3) -> set[str]:
        out: set[str] = set()
        for rec in self.records[-window:]:
            out |= set(rec.get("components") or [])
        return out

    def attempted_hypotheses(self, rounds: int = 10) -> list[str]:
        return [r.get("hypothesis", "") for r in self.records[-rounds:] if r.get("hypothesis")]

    def last_rounds(self, rounds: int) -> list[dict]:
        """최근 ``rounds``개 라운드의 기록을 기록 순서대로 돌려줍니다(R05·F05).

        한 라운드에는 후보 기록이 여러 개(m개) 있을 수 있으므로, 기록 개수가 아니라 서로 다른
        라운드를 셉니다. 라운드 식별자는 ``(run, t)``입니다. ``run``(실행 ID)이 있으면 다시 실행해
        ``t``가 겹쳐도 다른 라운드로 셉니다. ``run``이 없는 이전 기록은 ``t``만, ``t``도 없으면
        기록 하나를 한 라운드로 봅니다.
        """
        if rounds <= 0:
            return []
        seen: list = []
        keys = []
        for i, rec in enumerate(self.records):
            if "t" not in rec:
                key = ("i", i)
            else:
                key = ("t", rec.get("run") or "", rec["t"])
            keys.append(key)
            if key not in seen:
                seen.append(key)
        keep = set(seen[-rounds:])
        return [rec for rec, key in zip(self.records, keys) if key in keep]

    def to_prompt(self, rounds: int = 8) -> str:
        """제안자 프롬프트에 넣을 최근 ``rounds``개 라운드의 이력 요약."""
        lines = []
        for rec in self.last_rounds(rounds):
            verdict = "승격" if rec.get("accepted") else (rec.get("reason") or "탈락")[:60]
            run = f" run={rec['run']}" if rec.get("run") else ""
            lines.append(
                f"- t={rec.get('t')}{run} {rec.get('label', '')} "
                f"[{','.join(rec.get('components') or [])}] "
                f"dS={rec.get('delta_S') or 0:+.4f} dC={rec.get('delta_C') or 0:+.4f} → {verdict}"
            )
        return "\n".join(lines) if lines else "(이력 없음)"

    def recent_edits(self, rounds: int = 8, limit: int = 400) -> list[dict]:
        """제안자 프롬프트에 넣을 최근 ``rounds``개 라운드의 편집 원문(``edits``: 파일::교체 전->교체 후).

        요약 한 줄만으로는 왜 실패했는지 알 수 없어서 편집 내용을 함께 넘깁니다. 한 라운드의 후보가
        여럿이면 모두 넣습니다(``last_rounds``). 편집 하나는 ``limit``자에서 자릅니다(프롬프트 크기 상한).
        """
        out = []
        for rec in self.last_rounds(rounds):
            edits = [str(e) if len(str(e)) <= limit else str(e)[:limit] + "…(생략)"
                     for e in rec.get("edits") or []]
            if edits:
                out.append({"t": rec.get("t"), "run": rec.get("run") or "", "accepted": bool(rec.get("accepted")),
                            "reason": (rec.get("reason") or "")[:120], "edits": edits})
        return out
