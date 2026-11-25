"""도메인 인터페이스.

RRSI 코어는 "RAG"를 모릅니다. 도메인이 제공하는 것은 다음뿐입니다.

* ``evolve_ids`` / ``heldout_ids`` / ``smoke_ids`` — 평가할 문항
* ``run`` — 후보 하네스로 한 job을 실행
* ``score`` — 실행 산출물을 보상으로 채점 (심판은 도메인 쪽에만 있습니다)
* ``smoke`` — 평가 전 최소 생존 확인
* ``guards`` — 승격 금지 조건(도메인 지표)
* ``conditions`` / ``validity`` — 비교 조건과 평가 결과의 유효성(F01)

코어가 도메인 데이터를 읽지 않는다는 사실이 누출 방지의 구조적 근거입니다(8.2절).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskResult:
    """한 문항의 k회 트라이얼 보상과 토큰 사용량."""

    rewards: list[float] = field(default_factory=list)
    tokens: list[int | None] = field(default_factory=list)
    missing: int = 0

    @property
    def mean_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)

    @property
    def mean_tokens(self) -> float | None:
        vals = [t for t in self.tokens if t is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)


@dataclass
class EvalSummary:
    """한 job의 집계 결과.

    ``S``는 평균 보상, ``C``는 문항당 평균 정책 토큰입니다(7.4절의 S_hat, C_hat).
    """

    S: float
    C: float
    k: int
    per_task: dict[str, TaskResult] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    n: int = 0
    missing_frac: float = 0.0
    # 비교 조건(F01): 문항·반복 수·평가기·심판·고정 실행 조건. 후보가 바꿀 수 있는 하네스 값은 넣지 않습니다.
    conditions: dict[str, Any] = field(default_factory=dict)
    harness_tree: str = ""                 # 평가한 하네스 트리의 해시


class Domain:
    """도메인 구현이 만족해야 하는 최소 인터페이스."""

    name: str = "base"
    harness_path: str = "harness"          # 편집 가능한 유일한 디렉터리
    briefs: dict = {}
    critic_patterns: list[tuple[str, str]] = []
    component_signals: list[tuple[str, list[str]]] = []

    # --- 문항 -----------------------------------------------------------
    def evolve_ids(self) -> list[str]:
        raise NotImplementedError

    def heldout_ids(self) -> list[str]:
        raise NotImplementedError

    def smoke_ids(self, incumbent_per_task: dict | None = None) -> list[str]:
        return self.evolve_ids()[:2]

    # --- 실행과 채점 -----------------------------------------------------
    def run(self, root, runs_dir, job: str, ids: list[str], k: int,
            log_prefix: str = "") -> None:
        raise NotImplementedError

    def score(self, runs_dir, job: str, ids: list[str], k: int):
        raise NotImplementedError

    def smoke(self, root, runs_dir, job: str, ids: list[str]) -> tuple[bool, dict]:
        raise NotImplementedError

    # --- 승격 가드 -------------------------------------------------------
    def guards(self, incumbent: EvalSummary, candidate: EvalSummary) -> list[str]:
        return []

    # --- 비교 조건 (F01) --------------------------------------------------
    def conditions(self, root, ids: list[str], k: int) -> dict:
        """두 평가를 비교해도 되는 조건. 기본은 문항 목록과 반복 수뿐입니다."""
        return {"ids": sorted(ids), "k": int(k)}

    def validity(self, summary: EvalSummary) -> list[str]:
        """평가 결과를 기준선·비교·δ 보정에 써도 되는지. 문제가 있으면 사유 목록을 돌려줍니다."""
        return []

    # --- 편의 -----------------------------------------------------------
    def task_meta(self, task_id: str) -> dict:
        """제안자·분석가에게 보여 줄 문항 메타(정답 제외)."""
        return {"id": task_id}

    # --- 업스트림 분석 도구 호환 -----------------------------------------
    # google-research/rrsi 코어의 분석·보고 도구가 부르는 멤버입니다. 동봉 코어는 쓰지 않지만,
    # 이 어댑터를 업스트림 코어에 붙였을 때 속성 누락으로 멈추지 않도록 기본 동작을 둡니다.
    regression_threshold: float = 0.0

    def load_trial(self, runs_dir, job: str, task_id: str, trial: int) -> dict:
        """트라이얼 산출물 하나를 읽습니다(기본: runs/<도메인>/jobs/<job>/s<trial>/<task>.json)."""
        import json
        from pathlib import Path
        path = Path(runs_dir) / "jobs" / job / f"s{trial}" / f"{task_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def render_trace(self, trial: dict) -> str:
        """분석가에게 보여 줄 한 트라이얼 요약(기본: 답변·기권·오류만)."""
        return (f"answer={str(trial.get('answer', ''))[:200]!r} abstained={trial.get('abstained')} "
                f"error={trial.get('error')}")

    def task_row(self, task_id: str, result: TaskResult) -> dict:
        """보고 표의 한 행(기본: 문항 메타 + 평균 보상·토큰)."""
        return {**self.task_meta(task_id), "reward": result.mean_reward, "tokens": result.mean_tokens}
