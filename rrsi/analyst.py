"""실패·성공 트레이스 분석 (7.2절).

분석가는 정답 대신 실행 산출물(답변·기권·게이트 판정·토큰)만 봅니다.
정답을 보여 주면 제안자가 정답을 향해 코드를 고치게 되고, 그 순간 하네스는
평가셋에 과적합합니다.
"""
from __future__ import annotations

from pathlib import Path

from ._log import read_json


def _trials(runs_dir: Path, job: str) -> list[dict]:
    out = []
    for path in sorted((Path(runs_dir) / "jobs").glob(f"{job}/s[0-9]*/*.json")):
        data = read_json(path, {}) or {}
        if not isinstance(data, dict):
            continue                       # _ids.json 같은 목록 파일은 산출물에서 제외
        data.setdefault("task_id", path.stem)
        out.append(data)
    return out


def summarize_job(runs_dir: Path, job: str, k: int = 2) -> dict:
    trials = _trials(runs_dir, job)
    if not trials:
        return {"job": job, "n": 0, "note": "산출물 없음"}

    n = len(trials)
    abstained = [t for t in trials if t.get("abstained")]
    errors = [t for t in trials if t.get("error")]
    no_cite = [t for t in trials if not t.get("abstained") and not t.get("citations")]
    g2 = {}
    g1 = {}
    for t in trials:
        d = t.get("decisions") or {}
        g1[str((d.get("g1") or {}).get("choice"))] = g1.get(str((d.get("g1") or {}).get("choice")), 0) + 1
        verdict = str((d.get("g2") or {}).get("verdict"))
        g2[verdict] = g2.get(verdict, 0) + 1
    tokens = [t.get("usage", {}).get("total_tokens", 0) for t in trials]
    lat = sorted(t.get("latency_ms", 0) for t in trials)

    def head(items: list[dict], limit: int) -> list[dict]:
        return [{"task_id": t.get("task_id"),
                 "reason": (t.get("error") or ("abstained" if t.get("abstained") else
                            ("no_citation" if not t.get("citations") else ""))),
                 "g2": (t.get("decisions") or {}).get("g2", {}).get("verdict"),
                 "tokens": t.get("usage", {}).get("total_tokens")}
                for t in items[:limit]]

    return {
        "job": job, "n": n, "k": k,
        "abstain_rate": round(len(abstained) / n, 3),
        "error_rate": round(len(errors) / n, 3),
        "no_citation_rate": round(len(no_cite) / n, 3),
        "g1": g1, "g2": g2,
        "avg_tokens": round(sum(tokens) / n, 1),
        "p95_ms": lat[min(n - 1, int(round(0.95 * (n - 1))))],
        "failures": head(abstained + errors, 8),
        "samples": head([t for t in trials if not t.get("abstained") and t.get("citations")], 3),
    }


def feedback_text(summary: dict) -> str:
    """제안자에게 주는 피드백 F_t. 지표 이름과 방향만 알려 줍니다."""
    if not summary.get("n"):
        return "실행 산출물이 없습니다. 구성 오류부터 확인해 주세요."
    lines = [f"job {summary['job']}: n={summary['n']} "
             f"기권 {summary['abstain_rate']:.1%} 오류 {summary['error_rate']:.1%} "
             f"인용없음 {summary['no_citation_rate']:.1%}",
             f"평균 토큰 {summary['avg_tokens']:.0f}, p95 {summary['p95_ms']}ms",
             f"G1 분포 {summary['g1']}, G2 분포 {summary['g2']}"]
    if summary["failures"]:
        lines.append("실패 상위: " + ", ".join(str(f["task_id"]) for f in summary["failures"]))
    return "\n".join(lines)
