"""δ 재추정 (8.9절).

δ는 "같은 하네스를 여러 번 평가했을 때 점수가 흔들리는 폭"입니다. 후보의 상승이
이 폭보다 작으면 승격을 보류합니다. δ를 고정값으로 두면 모델·데이터가 바뀔 때
의미가 사라지므로, 같은 baseline을 여러 번 실행해 다시 측정합니다.

계산(점수의 순서와 무관합니다):
    σ        = 반복 점수의 표본 표준편차
    sd_null  = √2 · σ        (같은 하네스를 두 번 잰 점수 차이의 표준편차)
    δ        = z · sd_null

반복이 3회보다 적으면 σ를 믿을 수 없으므로 δ를 내지 않습니다(측정 불가).
이전 판은 모든 쌍의 한 방향 차이(a−b)에 표준편차를 적용해, 같은 점수라도 입력 순서에 따라
δ가 달라졌고 2회 실행이면 변동을 0으로 처리했습니다.
"""
from __future__ import annotations

import math
import statistics

MIN_RUNS = 3


def calibrate(scores: list[float], z: float = 2.0) -> dict:
    n = len(scores)
    if n < MIN_RUNS:
        return {"delta": None, "sd_null": None, "sigma": None, "z": z,
                "method": "sqrt2-sigma", "n": n,
                "note": f"baseline 반복 실행이 {MIN_RUNS}회 이상 필요합니다(측정 불가)"}
    sigma = statistics.stdev(scores)
    sd_null = math.sqrt(2.0) * sigma
    return {"delta": round(z * sd_null, 5), "sd_null": round(sd_null, 5), "sigma": round(sigma, 5),
            "z": z, "method": "sqrt2-sigma", "n": n}


def from_jobs(job_scores: dict[str, float], z: float = 2.0) -> dict:
    out = calibrate(list(job_scores.values()), z=z)
    out["scores"] = {k: round(v, 4) for k, v in job_scores.items()}
    return out


def job_problems(summary: dict, validity=None) -> list[str]:
    """eval.json 한 건을 δ 보정에 쓸 수 없는 사유(F02). 비어 있으면 씁니다.

    점수가 없거나 유한한 0~1 수가 아닌 job, 누락 트라이얼이 있는 job, 비교 조건·하네스 해시가
    없는 job(이전 판 eval.json)은 뺍니다. ``validity``는 도메인 검사(심판 장애·필수 값 누락)입니다.
    """
    if not isinstance(summary, dict) or not summary:
        return ["eval.json 없음"]
    out = []
    s = summary.get("S")
    if isinstance(s, bool) or not isinstance(s, (int, float)) or \
            (isinstance(s, float) and not math.isfinite(s)) or not 0.0 <= s <= 1.0:
        out.append(f"S={s!r}가 0~1의 유한한 수가 아닙니다")
    c = summary.get("C")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or \
            (isinstance(c, float) and not math.isfinite(c)) or c < 0:
        out.append(f"C={c!r}가 0 이상의 유한한 수가 아닙니다")
    for key in ("k", "n"):
        val = summary.get(key)
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            out.append(f"{key}={val!r}가 양의 정수가 아닙니다")
    mf = summary.get("missing_frac")
    if isinstance(mf, bool) or mf != 0:
        out.append(f"누락 트라이얼 missing_frac={mf!r}")
    # 트라이얼 수(G02)가 있으면 산출물·비용 측정이 예상 트라이얼과 모두 같아야 합니다.
    exp = summary.get("trials_expected")
    if exp is not None and not (summary.get("trials_done") == exp == summary.get("cost_trials")):
        out.append(f"트라이얼 {summary.get('trials_done')!r}/{exp!r}, "
                   f"비용 측정 {summary.get('cost_trials')!r}/{exp!r}")
    if not summary.get("conditions") or not summary.get("harness_tree"):
        out.append("비교 조건·하네스 해시 없음(이전 판 eval.json)")
    if validity is not None and not out:
        out += list(validity(summary))
    return out


def fingerprint_of(summary: dict) -> dict:
    """δ를 잰 조건의 지문: 하네스 해시 + 비교 조건(문항·k·평가기·심판·고정 실행 조건)."""
    return {"harness_tree": summary.get("harness_tree"), "conditions": summary.get("conditions")}


def from_summaries(summaries: dict[str, dict], z: float = 2.0, validity=None) -> dict:
    """job별 eval.json으로 δ를 잽니다(F02).

    쓸 수 없는 job(``job_problems``)은 빼고, 남은 job은 모두 같은 지문(같은 하네스·문항·반복 수·
    평가기·심판·실행 조건)이어야 합니다. 지문이 섞이면 δ를 내지 않습니다. 유효한 job이 3개보다
    적어도 내지 않습니다. 결과에는 지문과 제외 사유를 남겨, 재사용할 때 조건을 대조합니다.
    """
    excluded = {job: p for job, s in summaries.items() if (p := job_problems(s, validity))}
    valid = {job: s for job, s in summaries.items() if job not in excluded}
    fps = {json_key(fingerprint_of(s)) for s in valid.values()}
    if len(fps) > 1:
        out = calibrate([], z=z)
        out["note"] = "job마다 하네스·문항·반복 수·평가기·심판·실행 조건이 다릅니다(측정 불가)"
    else:
        out = calibrate([float(s["S"]) for s in valid.values()], z=z)
        if valid:
            out["fingerprint"] = fingerprint_of(next(iter(valid.values())))
    out["scores"] = {job: round(float(s["S"]), 4) for job, s in valid.items()}
    out["excluded"] = excluded
    return out


def json_key(obj) -> str:
    import json
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
