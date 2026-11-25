#!/usr/bin/env python3
"""임계값 검사 (9.7절).

    python eval/check_thresholds.py /tmp/report.json \
        --max-correct-drop 0.01 --max-cite-drop 0.02 --max-abstain-drop 0.03 \
        --max-error-rise 0.01 --item-regressions

기준선 대비 하락이 허용치를 넘으면 exit 1이고, CI 작업이 실패합니다. 병합까지 막으려면 기본 브랜치의
보호 규칙 또는 Ruleset에서 이 작업(rag-regression의 job ``eval``)을 필수 상태 검사로 지정해야 합니다.
기준선이 없거나, 필수 지표가 빠졌거나, 표본 수가 모자라도 exit 1입니다(조용히 통과하지 않음).
처음 한 번은 --init-baseline으로 현재 보고서를 기준선으로 등록합니다(사람이 검토한 뒤 커밋).
문항 구성·골든셋·평가기를 일부러 바꿨으면 --update-baseline으로 기준선을 새로 등록합니다.

성능 차이를 계산하기 전에 현재 보고서와 기준선을 **둘 다** 같은 규칙으로 검증합니다.

* ``n``은 정수여야 하고(문자열 "12"나 12.0은 거절) ``--min-n`` 이상이어야 합니다.
* 다섯 지표(정답률·인용 정밀도·두 기권율·실행 오류율)는 0~1 범위의 수여야 합니다.
* ``ids``(실제로 평가한 문항)는 중복이 없고 개수가 ``n``과 같아야 하며, ``subset_sha256``은
  그 목록의 해시와 같아야 합니다(E06).
* ``items``(문항별 결과)는 반드시 있어야 하고, 개수·ID·순서가 ``ids``와 같아야 합니다. 문항마다
  필드 자료형과 값 범위, ``outcome``·``abstained``·``reason``·``error``의 일관성(domains/rag/outcomes.py
  분류표로 다시 계산)을 확인하고, 문항별 결과로 집계 지표를 다시 계산해 보고서 값과 대조합니다(R02).
* 심판 장애(``judge_errors``)가 한 건이라도 있으면 정답률을 믿을 수 없으므로 실패입니다(R03).
* ``goldenset_version``·``evaluator``·``judge``·``subset_sha256``이 두 보고서에서 같아야 합니다.
  다르면 "비교할 수 없음"으로 끝냅니다(exit 1). 문항이나 채점 방식이 다른 두 수의 차이는
  성능 변화가 아니기 때문입니다.

--init-baseline·--update-baseline도 같은 검증을 통과한 보고서만 기준선으로 등록합니다.
기권은 두 방향을 확인합니다. 답할 수 없는 문항의 기권율(기권 정확도)은 유지되거나 올라야 하고,
답할 수 있는 문항의 기권율(과잉 기권)은 유지되거나 내려가야 합니다. 둘 다 --max-abstain-drop을 씁니다.
기권율은 기권으로 인정하는 결과(정상 기권·보안 차단)만 세고, 실행 오류·운영 장애·예산 초과·알 수
없는 사유로 끝난 결과는 error_rate(실패율)로 따로 봅니다(E05·R01).

집계 지표는 문항끼리 상쇄될 수 있습니다(한 문항이 새로 틀리고 다른 문항이 새로 맞으면 정답률은
그대로입니다). --item-regressions를 켜면 기준선에서 통과한 문항이 이번에 실패한 경우를 문항별로
찾아 하나라도 있으면 실패합니다. 통과의 뜻: 답변 가능 문항은 실패 없이 correct=1, 답변 불가
문항은 기권으로 인정하는 결과(정상 기권·보안 차단)입니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "domains" / "rag"))
# 회귀 평가·진화 채점과 같은 분류표를 씁니다(R01). 외부 패키지 없이 import됩니다.
from outcomes import FAILURES, OUTCOMES, REFUSALS, outcome  # noqa: E402

METRICS = ("correct", "cite_precision", "abstain_unanswerable", "abstain_answerable", "error_rate")
VERSION_KEYS = ("goldenset_version", "evaluator", "judge", "subset_sha256")
JUDGE_STATES = ("ok", "error", "skipped")
TOL = 5e-4                      # 집계값 대조 허용 오차(보고서는 소수 4자리로 반올림합니다)


def get(report: dict, key: str) -> float | None:
    val = report.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    if isinstance(val, int):                    # 아주 큰 정수는 float로 바꿀 수 없습니다(RT11)
        return float(val) if abs(val) <= 2 ** 53 else None
    return val if math.isfinite(val) else None


def subset_sha256(ids: list[str]) -> str:
    """eval/run_subset.py와 같은 정규화(중복 제거·정렬) 해시."""
    norm = json.dumps(sorted(set(ids)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def validate(report, name: str, min_n: int) -> list[str]:
    """보고서 한 건의 형식 오류 목록을 돌려줍니다. 비어 있으면 유효합니다."""
    if not isinstance(report, dict):
        return [f"{name}: 보고서가 JSON 객체가 아닙니다"]
    errs = []
    n = report.get("n")
    n_ok = isinstance(n, int) and not isinstance(n, bool)
    if not n_ok:
        errs.append(f"{name}: n이 없거나 정수가 아닙니다({n!r})")
    elif n < min_n:
        errs.append(f"{name}: 표본 수 {n} < 최소 {min_n}")
    for key in METRICS:
        val = get(report, key)
        if val is None:
            errs.append(f"{name}: {key} 지표가 없거나 수가 아닙니다({report.get(key)!r})")
        elif not 0.0 <= val <= 1.0:
            errs.append(f"{name}: {key}={val}이 0~1 범위를 벗어났습니다")
    for key in VERSION_KEYS:
        val = report.get(key)
        if not isinstance(val, str) or not val.strip():
            errs.append(f"{name}: {key}가 없습니다")
    ids = report.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        errs.append(f"{name}: ids(평가한 문항 목록)가 없습니다")
        return errs
    if len(set(ids)) != len(ids):
        errs.append(f"{name}: ids에 중복 문항이 있습니다(고유 {len(set(ids))}, 전체 {len(ids)})")
    if n_ok and len(ids) != n:
        errs.append(f"{name}: ids 개수 {len(ids)}와 n={n}이 다릅니다")
    nu = report.get("n_unique")
    if isinstance(nu, bool) or not isinstance(nu, int) or nu != len(set(ids)):
        errs.append(f"{name}: n_unique={nu!r}가 ids의 고유 문항 수 {len(set(ids))}와 다릅니다")
    if isinstance(report.get("subset_sha256"), str) and report["subset_sha256"] != subset_sha256(ids):
        errs.append(f"{name}: subset_sha256이 ids와 맞지 않습니다")
    errs += validate_items(report, name, ids)
    return errs


def _num01(v) -> bool:
    """0~1 범위의 수인가. 참/거짓은 거절하고, 정수는 범위부터 비교합니다(F07).

    ``math.isfinite(10**400)``은 float 변환에서 OverflowError를 내므로, 정수는 isfinite를 부르지
    않고 범위만 봅니다(정수는 항상 유한합니다). 실수는 NaN·무한대를 거절합니다.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    if isinstance(v, int):
        return 0 <= v <= 1
    return math.isfinite(v) and 0.0 <= v <= 1.0


def row_errors(r) -> list[str]:
    """문항별 결과 한 줄의 자료형·값 범위·일관성 오류(R02)."""
    if not isinstance(r, dict):
        return ["객체가 아닙니다"]
    e = []
    if not isinstance(r.get("id"), str):
        e.append(f"id가 문자열이 아닙니다({r.get('id')!r})")
    for key in ("answerable", "abstained"):
        if not isinstance(r.get(key), bool):
            e.append(f"{key}가 참/거짓이 아닙니다({r.get(key)!r})")
    if r.get("outcome") not in OUTCOMES:
        e.append(f"outcome={r.get('outcome')!r}이 허용 값이 아닙니다")
    if not isinstance(r.get("reason"), str):
        e.append("reason이 문자열이 아닙니다")
    err = r.get("error")
    if err is not None and (not isinstance(err, str) or not err.strip()):
        e.append(f"error는 없음(null) 또는 비어 있지 않은 문자열이어야 합니다({err!r})")
    if not _num01(r.get("reward")):
        e.append(f"reward={r.get('reward')!r}가 0~1의 수가 아닙니다")
    cp = r.get("cite_precision")
    if cp is not None and not _num01(cp):
        e.append(f"cite_precision={cp!r}가 0~1의 수가 아닙니다")
    if r.get("judge") not in JUDGE_STATES:
        e.append(f"judge={r.get('judge')!r}가 허용 값이 아닙니다")
    if e:
        return e
    # 일관성: outcome은 abstained·reason·error에서 다시 계산한 분류와 같아야 합니다.
    kind = outcome(r)
    if r["outcome"] != kind:
        e.append(f"outcome={r['outcome']}이 abstained·reason·error로 계산한 {kind}와 다릅니다")
    c = r.get("correct")
    if r["answerable"]:
        if r["judge"] == "error":
            if c is not None:
                e.append("심판 장애(judge=error)인데 correct가 있습니다")
        elif isinstance(c, bool) or c not in (0.0, 1.0):
            e.append(f"correct={c!r}가 0 또는 1이 아닙니다")
        if kind != "answered" and (r["judge"] != "skipped" or c != 0.0):
            e.append("답하지 않은 결과인데 정답 판정을 했거나 correct가 0이 아닙니다")
        elif kind not in FAILURES and (c is None or (not isinstance(c, bool) and c in (0.0, 1.0))):
            # correct가 0·1·없음일 때만 다시 계산합니다(큰 정수를 곱하면 OverflowError, F07).
            # 8.4절 보상식으로 다시 계산(cite_precision은 소수 4자리 반올림 값)
            expect = min(1.0, 0.65 * (c or 0.0) + 0.20 * (cp or 0.0)
                         + 0.15 * (kind == "answered"))
            if abs(r["reward"] - expect) > TOL:
                e.append(f"reward={r['reward']}가 correct·cite_precision으로 계산한 "
                         f"{expect:.4f}와 다릅니다")
    else:
        if c is not None or r["judge"] != "skipped":
            e.append("답변 불가 문항에 정답 판정 값이 있습니다")
        expect = 1.0 if kind in REFUSALS else 0.0
        if r["reward"] != expect:
            e.append(f"답변 불가 문항 reward={r['reward']}가 규칙 값 {expect}와 다릅니다")
    if kind in FAILURES and r["reward"] != 0.0:
        e.append(f"실패 결과({kind})인데 reward={r['reward']}입니다")
    return e


def recompute(rows: list[dict]) -> dict:
    """문항별 결과로 집계 지표를 다시 계산합니다(eval/run_subset.py와 같은 정의)."""
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def rate(rs, kinds):
        return sum(r["outcome"] in kinds for r in rs) / len(rs) if rs else None

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {
        "S": mean([r["reward"] for r in rows]),
        "correct": mean([r["correct"] for r in ans]),
        "cite_precision": mean([r.get("cite_precision") for r in ans]),
        "abstain_answerable": rate(ans, REFUSALS) if ans else 0.0,
        "abstain_unanswerable": rate(unans, REFUSALS),
        "error_rate": rate(rows, FAILURES) if rows else 0.0,
        "judge_errors": sum(r["judge"] == "error" for r in rows),
        "outcomes": {k: sum(r["outcome"] == k for r in rows) for k in OUTCOMES},
    }


def validate_items(report: dict, name: str, ids: list[str]) -> list[str]:
    """``items``(문항별 결과)가 있고 ``ids``·집계 지표와 맞는지 확인합니다(R02)."""
    rows = report.get("items")
    if not isinstance(rows, list) or not rows:
        return [f"{name}: items(문항별 결과)가 없거나 비어 있습니다"]
    errs, bad = [], 0
    for i, r in enumerate(rows):
        for msg in row_errors(r):
            bad += 1
            if bad <= 10:
                rid = r.get("id") if isinstance(r, dict) else None
                errs.append(f"{name}: items[{i}]({rid}) {msg}")
    item_ids = [r.get("id") if isinstance(r, dict) else None for r in rows]
    if len(set(map(str, item_ids))) != len(item_ids):
        errs.append(f"{name}: items에 중복 문항이 있습니다")
    if len(rows) != len(ids):
        errs.append(f"{name}: items 개수 {len(rows)}와 ids 개수 {len(ids)}가 다릅니다")
    elif item_ids != ids:
        what = "순서" if sorted(map(str, item_ids)) == sorted(ids) else "ID 집합"
        errs.append(f"{name}: items의 {what}가 ids와 다릅니다")
    if errs:
        return errs
    # 집계 지표 대조: 문항별 결과와 다른 집계는 손으로 고쳤거나 다른 실행의 값입니다.
    for key, val in recompute(rows).items():
        got = report.get(key)
        if key in ("outcomes", "judge_errors"):
            if got != val:
                errs.append(f"{name}: {key}={got!r}가 문항별 결과로 계산한 {val!r}와 다릅니다")
        elif val is None or get(report, key) is None:
            if not (val is None and got is None):
                errs.append(f"{name}: {key}={got!r}가 문항별 결과로 계산한 {val!r}와 다릅니다")
        elif abs(get(report, key) - val) > TOL:
            errs.append(f"{name}: {key}={got}가 문항별 결과로 계산한 {val:.4f}와 다릅니다")
    if report.get("judge_errors"):
        errs.append(f"{name}: 심판 장애 {report['judge_errors']}건 — 정답률을 믿을 수 없습니다"
                    "(심판을 복구한 뒤 다시 평가하십시오)")
    return errs


def item_passed(row: dict) -> bool:
    """문항 하나의 통과 여부: 답변 가능은 실패 없이 정답, 답변 불가는 기권으로 인정하는 결과."""
    kind = row.get("outcome")
    if kind in FAILURES or kind not in OUTCOMES:
        return False
    if row.get("answerable"):
        return row.get("correct") == 1.0
    return kind in REFUSALS


def item_regressions(report: dict, base: dict) -> list[str]:
    """기준선에서 통과했는데 이번에 실패한 문항 ID 목록. 두 보고서는 validate()를 통과해야 합니다."""
    now = {r.get("id"): r for r in report.get("items") or [] if isinstance(r, dict)}
    out = []
    for row in base.get("items") or []:
        if isinstance(row, dict) and item_passed(row):
            cur = now.get(row.get("id"))
            if cur is None or not item_passed(cur):
                out.append(str(row.get("id")))
    return out


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"읽을 수 없습니다({exc})"


def register(report, base_path: Path, errs: list[str], old=None) -> int:
    """검증을 통과한 보고서만 기준선으로 씁니다."""
    if errs:
        for e in errs:
            print(f"[thresholds] 실패: {e}")
        print("[thresholds] 실패: 기준선으로 등록하지 않았습니다")
        return 1
    if isinstance(old, dict):
        for k in VERSION_KEYS:
            if old.get(k) != report[k]:
                print(f"[thresholds] 기준선 {k}: {old.get(k)} → {report[k]}")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[thresholds] 기준선 등록: {base_path} — 검토 후 커밋하십시오")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--baseline", default="eval/reports/base_ragas.json")
    ap.add_argument("--max-correct-drop", type=float, default=0.01)
    ap.add_argument("--max-cite-drop", type=float, default=0.02)
    ap.add_argument("--max-abstain-drop", type=float, default=0.03)
    ap.add_argument("--max-error-rise", type=float, default=0.01,
                    help="실패율(error_rate: 실행 오류·운영 장애·예산 초과·알 수 없는 사유) 증가 허용치")
    ap.add_argument("--min-n", type=int, default=10, help="보고서의 최소 문항 수(n)")
    ap.add_argument("--item-regressions", action="store_true",
                    help="기준선에서 통과한 문항이 하나라도 실패하면 실패(집계 지표의 상쇄 방지)")
    ap.add_argument("--init-baseline", action="store_true",
                    help="기준선이 없을 때 현재 보고서를 기준선으로 등록하고 종료")
    ap.add_argument("--update-baseline", action="store_true",
                    help="문항·골든셋·평가기를 일부러 바꾼 뒤 기준선을 새로 등록하고 종료")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # 윈도 cp949 콘솔에서 —·→ 출력 때문에 중단되지 않도록
        sys.stdout.reconfigure(errors="replace")

    report = load(Path(a.report))
    errs = validate(report, "현재 보고서", a.min_n)
    base_path = Path(a.baseline)
    if a.update_baseline:
        return register(report, base_path, errs, load(base_path) if base_path.exists() else None)
    if not base_path.exists():
        if a.init_baseline:
            return register(report, base_path, errs)
        print(f"[thresholds] 실패: 기준선 없음({base_path}). 처음이면 --init-baseline으로 등록하십시오")
        return 1
    base = load(base_path)
    errs += validate(base, "기준선", a.min_n)
    if errs:
        for e in errs:
            print(f"[thresholds] 실패: {e}")
        return 1
    # 버전·문항 목록이 다르면 차이를 계산하지 않습니다. 문항·채점 방식이 다른 수는 비교 대상이 아닙니다.
    mismatched = [k for k in VERSION_KEYS if report[k] != base[k]]
    if mismatched:
        for k in mismatched:
            print(f"[thresholds] 비교할 수 없음: {k} 불일치(보고서 {report[k]}, 기준선 {base[k]})")
        print("[thresholds] 실패: 같은 골든셋·평가기·문항으로 다시 평가하거나, 의도한 변경이면 "
              "--update-baseline으로 기준선을 새로 등록하십시오")
        return 1

    # (키, 표시 이름, 허용치, 방향) — 방향 +1: 높을수록 좋음, -1: 낮을수록 좋음
    checks = [
        ("correct", "correct", a.max_correct_drop, +1),
        ("cite_precision", "cite_precision", a.max_cite_drop, +1),
        ("abstain_unanswerable", "abstain(답변불가, 기권 정확도)", a.max_abstain_drop, +1),
        ("abstain_answerable", "abstain(답변가능, 과잉 기권)", a.max_abstain_drop, -1),
        ("error_rate", "error_rate(실패: 오류·장애·예산)", a.max_error_rise, -1),
    ]
    failed = False
    for key, label, tol, direction in checks:
        cur, ref = get(report, key), get(base, key)
        if cur is None or ref is None:
            print(f"[thresholds] {label}: 지표 없음(보고서 {cur}, 기준선 {ref}) FAIL")
            failed = True
            continue
        worse = (ref - cur) * direction          # 나빠진 양(양수면 악화)
        ok = worse <= tol
        print(f"[thresholds] {label}: base {ref:.4f} → now {cur:.4f} "
              f"(d {cur - ref:+.4f}, 악화 허용 {tol:.3f}) {'ok' if ok else 'FAIL'}")
        failed = failed or not ok

    if a.item_regressions:
        lost = item_regressions(report, base)
        print(f"[thresholds] 문항별 퇴행: {len(lost)}건"
              + (f" ({', '.join(lost[:10])}) FAIL" if lost else " ok"))
        failed = failed or bool(lost)

    print("[thresholds] " + ("실패: 기준선 대비 하락이 허용치를 넘었습니다" if failed else "통과"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
