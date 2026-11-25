#!/usr/bin/env python3
"""PR마다 실행하는 평가 부분집합 (9.7절). 야간 평가(10.8절)도 같은 스크립트를 씁니다.

    python eval/run_subset.py --subset eval/ci/regression_subset.json --out /tmp/report.json

하네스를 같은 프로세스 안에서 실행합니다. 배포된 서비스가 아니라 저장소의 코드를 평가하는
것이 목적이므로 클러스터 없이 동작합니다. 기준선과의 비교는 eval/check_thresholds.py가 맡습니다.

문항 목록은 실행 전에 검사합니다(E06). 빈 목록, 중복 ID, 골든셋에 없는 ID가 있으면 평가하지
않고 종료 코드 2로 끝냅니다. 보고서에는 실제로 평가한 ID 목록(``ids``), 고유 문항 수
(``n_unique``), 정렬한 ID 목록의 해시(``subset_sha256``)를 남겨 기준선과 같은 문항인지 가립니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "domains" / "rag"))
sys.path.insert(0, str(ROOT))

from harness.llm import DecisionClient  # noqa: E402
from harness.pipeline import RAGPipeline  # noqa: E402
from harness.retrieve import chunk_doc, load_corpus  # noqa: E402
from judge import (EVALUATOR_VERSION, JUDGE_ERROR, build_judge,  # noqa: E402
                   goldenset_version, judge_answer, judge_id, score_trial, trial_tokens)
from outcomes import FAILURES, OUTCOMES, REFUSALS  # noqa: E402

DATA = ROOT / "domains" / "rag" / "data"


class SubsetError(ValueError):
    """평가할 문항 목록이 잘못되었습니다(빈 목록·중복·없는 ID)."""


def load_goldenset() -> dict[str, dict]:
    out = {}
    for line in (DATA / "goldenset_v1.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            out[item["id"]] = item
    return out


def chunk_texts() -> dict[str, str]:
    texts = {}
    for doc in load_corpus(DATA / "corpus"):
        for p in chunk_doc(doc):
            texts[p.chunk_id] = p.text
    return texts


def load_subset(path: Path) -> list:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return list(obj.get("ids") or obj.get("evolve") or [])
    return list(obj) if isinstance(obj, list) else []


def validate_ids(ids: list, items: dict) -> list[str]:
    """문항 목록을 검사해 그대로 돌려줍니다. 문제가 있으면 ``SubsetError``입니다(E06).

    없는 ID를 조용히 버리거나 중복을 그대로 두면, 같은 이름의 부분집합이라도 실제로 평가한
    문항이 기준선과 달라집니다. 그런 보고서는 비교가 성립하지 않습니다.
    """
    if not ids:
        raise SubsetError("문항 목록이 비어 있습니다")
    bad = [i for i in ids if not isinstance(i, str) or not i.strip()]
    if bad:
        raise SubsetError(f"문자열이 아닌 ID가 있습니다: {bad[:5]!r}")
    seen, dup = set(), []
    for i in ids:
        if i in seen and i not in dup:
            dup.append(i)
        seen.add(i)
    if dup:
        raise SubsetError(f"중복 ID {len(dup)}개: {', '.join(dup[:10])}")
    unknown = [i for i in ids if i not in items]
    if unknown:
        raise SubsetError(f"골든셋에 없는 ID {len(unknown)}개: {', '.join(unknown[:10])}")
    return list(ids)


def subset_sha256(ids: list[str]) -> str:
    """정규화(중복 제거·정렬)한 문항 목록의 해시 앞 16자리. 순서가 달라도 같은 값입니다."""
    norm = json.dumps(sorted(set(ids)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def p95(vals: list[float]) -> float:
    """최근접 순위 p95. 표본이 20개 미만이면 사실상 최댓값입니다."""
    if not vals:
        return 0
    xs = sorted(vals)
    return xs[max(0, math.ceil(0.95 * len(xs)) - 1)]


def evaluate(ids: list[str], items: dict, texts: dict, pipe, judge, subset: str = "") -> dict:
    """문항마다 하네스를 실행하고 채점해 보고서를 만듭니다. ``ids``는 검사를 마친 목록입니다."""
    rows = []
    unknown_calls = 0                 # 미측정 호출 수(있으면 tokens는 추정치로 채운 값입니다)
    for tid in ids:
        item = items[tid]
        res = pipe.answer_sync(item["question"], item["tenant"], "pro")
        unknown_calls += trial_tokens(res.get("usage"))[1]
        # 진화 채점(adapter._reward)과 같은 함수입니다. 정답 판정은 문항당 한 번이고(R03),
        # 실패 결과(운영 장애·예산 초과·알 수 없는 사유·실행 오류)는 기권이어도 0점입니다(E05·R01).
        sc = score_trial(item, res, texts, lambda answer, it=item: judge_answer(judge, it, answer))
        kind, reward, ok, total = sc["outcome"], sc["reward"], sc["ok"], sc["total"]
        rows.append({
            "id": tid, "type": item["type"], "answerable": item["answerable"],
            # outcome: outcomes.py의 분류(answered·abstained·blocked·ops_failure·budget_exceeded·
            # unknown·error). 기권으로 인정하는 것은 abstained·blocked뿐입니다.
            "outcome": kind,
            "abstained": res["abstained"], "reason": res.get("reason", ""),
            "error": res.get("error"),
            "reward": round(reward, 4),
            # judge: ok(판정함) · error(심판 장애, correct=None) · skipped(판정 불필요)
            "correct": sc["correct"] if item["answerable"] else None,
            "judge": sc["judge"],
            "cite_precision": round(ok / total, 4) if total else None,
            # 미측정 호출은 0이 아니라 글자 수 추정치로 채웁니다(RT06).
            "tokens": trial_tokens(res.get("usage"))[0],
            "latency_ms": res.get("latency_ms", 0),
            "answer": res.get("answer", ""),
            # RAGAS(eval/ragas_metrics.py)에 넘길 문맥. 파이프라인이 리랭킹 후 순서의
            # context_ids를 주면 그것을, 없으면 검색 결과 순서(retrieved)를 씁니다.
            "contexts": [texts[c] for c in (res.get("context_ids") or res.get("retrieved") or [])
                         if c in texts],
        })

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    def rate(rs, kinds):
        return round(sum(r["outcome"] in kinds for r in rs) / len(rs), 4) if rs else None

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    report = {
        "subset": str(subset), "n": len(rows),
        # 비교 가능 조건: 골든셋·평가기·심판·문항 해시가 기준선과 같아야 check_thresholds.py가 차이를 계산합니다.
        "goldenset_version": goldenset_version(DATA / "goldenset_v1.jsonl"),
        "evaluator": EVALUATOR_VERSION,
        # 실제로 쓴 심판(mode/model/주소 표식/판정 버전). 기준선과 다르면 비교하지 않습니다(R03·F08).
        "judge": judge_id(judge),
        "subset_sha256": subset_sha256(ids),
        "n_unique": len(set(ids)),
        "ids": list(ids),
        "S": mean([r["reward"] for r in rows]),
        # 심판 장애 문항(correct=None)은 평균에서 빠지고 judge_errors로 셉니다. 한 건이라도 있으면
        # check_thresholds.py가 보고서를 믿을 수 없다고 보고 실패합니다(R03).
        "correct": mean([r["correct"] for r in answerable]),
        "judge_errors": sum(r["judge"] == JUDGE_ERROR for r in rows),
        "cite_precision": mean([r["cite_precision"] for r in answerable]),
        # 기권율은 기권으로 인정하는 결과(정상 기권·보안 차단)만 셉니다. 운영 장애·예산 초과·알 수
        # 없는 사유·실행 오류는 error_rate(실패율)로 따로 셉니다(E05·R01).
        "abstain_answerable": rate(answerable, REFUSALS) if answerable else 0.0,
        "abstain_unanswerable": rate(unanswerable, REFUSALS),
        "error_rate": rate(rows, FAILURES) if rows else 0.0,
        "outcomes": {k: sum(r["outcome"] == k for r in rows) for k in OUTCOMES},
        "tokens": mean([r["tokens"] for r in rows]),
        "usage_unknown_calls": unknown_calls,
        "p95_ms": p95([r["latency_ms"] for r in rows]),
        "by_type": {},
        "items": rows,
    }
    for t in sorted({r["type"] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        report["by_type"][t] = {"n": len(rs), "S": mean([r["reward"] for r in rs])}
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="eval/ci/regression_subset.json")
    ap.add_argument("--out", default="/tmp/report.json")
    ap.add_argument("--config", default="domains/rag/harness/config.yaml")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # 윈도 cp949 콘솔에서 출력 때문에 중단되지 않도록
        sys.stdout.reconfigure(errors="replace")

    items = load_goldenset()
    try:
        ids = validate_ids(load_subset(Path(a.subset)), items)
    except (SubsetError, OSError, ValueError) as exc:
        print(f"[run_subset] 실패: {a.subset}: {exc}")
        return 2
    # 심판은 진화 채점과 같은 rrsi.json의 judge_* 설정으로 만듭니다(F08).
    judge_cfg = json.loads((ROOT / "domains" / "rag" / "rrsi.json").read_text(encoding="utf-8"))
    report = evaluate(ids, items, chunk_texts(), RAGPipeline.from_config(a.config),
                      build_judge(judge_cfg), subset=a.subset)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("n", "S", "correct", "cite_precision", "abstain_answerable",
                       "abstain_unanswerable", "error_rate", "judge_errors", "tokens")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
