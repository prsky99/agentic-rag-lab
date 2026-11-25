"""채점 유틸 — 정답 판정과 인용 지지 검사 (8.5절).

심판은 도메인 쪽에만 두고, 하네스의 import 대상에서 제외합니다.
"""
from __future__ import annotations

import math
import re

try:                                     # 패키지로 import될 때
    from harness.llm import DecisionClient, DecisionUnavailable
    from outcomes import FAILURES, REFUSALS, outcome  # noqa: F401  (분류표, R01)
except ImportError:                      # domains/rag가 sys.path에 있을 때
    from .harness.llm import DecisionClient, DecisionUnavailable
    from .outcomes import FAILURES, REFUSALS, outcome  # noqa: F401


# 채점·집계 방식의 버전. 보상 계산이나 비용(C) 집계가 바뀌면 올립니다. 버전이 다른 두
# 보고서는 비교하지 않습니다(eval/check_thresholds.py, 평가 지문).
# rag-eval/3: 미측정 사용량을 0으로 세지 않고 추정치로 채웁니다(RT06).
# rag-eval/4: 실행 오류(error)가 있는 트라이얼은 기권 여부와 무관하게 0점이고, 정상 기권으로
#             세지 않습니다. 회귀 평가(run_subset)와 진화 채점(adapter)이 같은 함수를 씁니다(E05).
# rag-eval/5: 기권 사유를 분류해 운영 장애·예산 초과·알 수 없는 사유는 0점·실패로 셉니다(R01).
#             정답 판정은 문항·답변당 한 번이고, 심판 장애는 오답과 구별해 기록합니다(R03).
EVALUATOR_VERSION = "rag-eval/5"

# 정답 판정 방식의 버전(질문 문구·0.80 문턱). 바꾸면 올립니다. 보고서의 ``judge`` 표식에 들어갑니다.
JUDGE_VERSION = "same-v1"

# 심판 상태: ok(판정함) · error(심판 장애·형식 오류, 오답과 구별) · skipped(판정할 필요 없음)
JUDGE_OK, JUDGE_ERROR, JUDGE_SKIPPED = "ok", "error", "skipped"


def goldenset_version(path) -> str:
    """골든셋 버전 표식: 파일 이름 + 내용 해시 앞 12자리."""
    import hashlib
    from pathlib import Path
    path = Path(path)
    return f"{path.stem}@sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


def trial_tokens(usage) -> tuple[int | None, int]:
    """트라이얼 한 건의 비용 토큰과 미측정 호출 수를 돌려줍니다(RT06).

    미측정 호출은 합계(``total_tokens``)에 0으로 들어 있으므로, 그대로 쓰면 비용이 실제보다
    작아집니다. 그래서 글자 수 추정치(``unknown_estimate_tokens``)를 더합니다. 사용량 기록이
    아예 없는 트라이얼(실행 오류)은 ``None``을 돌려주어 평균에서 뺍니다.
    """
    if not isinstance(usage, dict) or not usage:
        return None, 0

    def n(key: str) -> int:
        v = usage.get(key)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else 0

    return n("total_tokens") + n("unknown_estimate_tokens"), n("unknown_calls")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def span_supported(span: str, chunk_text: str) -> bool:
    """인용 구간이 실제 청크 본문에 있는가. 줄바꿈만 다르면 같은 텍스트로 봅니다."""
    return bool(span) and bool(chunk_text) and norm(span) in norm(chunk_text)


def citation_precision(citations: list[dict], retrieved: list[str],
                       chunk_texts: dict[str, str]) -> tuple[int, int]:
    ok = total = 0
    rset = set(retrieved or [])
    for c in citations or []:
        total += 1
        cid = c.get("chunk_id")
        if cid in rset and span_supported(c.get("span") or "", chunk_texts.get(cid or "", "")):
            ok += 1
    return ok, total




def endpoint_id(url: str) -> str:
    """주소의 비교용 표식(F08): 스킴·호스트·포트·경로의 SHA-256 앞 12자리.

    사용자 정보(``user:pass@``)·질의 문자열·조각처럼 비밀이 들어갈 수 있는 부분은 뺍니다.
    같은 모델 이름이라도 주소가 다르면 다른 표식이 됩니다. 주소가 없으면 ``-``입니다.
    """
    import hashlib
    from urllib.parse import urlsplit
    url = (url or "").strip()
    if not url:
        return "-"
    try:
        p = urlsplit(url)
        host = (p.hostname or "").lower() + (f":{p.port}" if p.port else "")
        norm = f"{p.scheme.lower()}://{host}{p.path.rstrip('/')}"
    except ValueError:
        norm = url
    return "ep-" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def build_judge(cfg: dict) -> DecisionClient:
    """``rrsi.json``의 심판 설정으로 정답 판정 심판을 만듭니다(F08).

    ``judge_mode``(auto|remote|local|offline), ``judge_model``, ``judge_base_url``, ``judge_revision``을
    읽습니다. 빈 값은 환경 변수(``DECISION_MODE``·``DECISION_MODEL``·``DECISION_BASE_URL``)를 따르고,
    auto는 주소가 있으면 remote, 없으면 local입니다. 실제로 정해진 값은 ``judge_id``에 남습니다.
    """
    cfg = cfg or {}
    client = DecisionClient(base_url=(cfg.get("judge_base_url") or None),
                            model=(cfg.get("judge_model") or None),
                            mode=str(cfg.get("judge_mode") or "auto"))
    client.revision = str(cfg.get("judge_revision") or "")
    return client


def judge_id(client: DecisionClient) -> str:
    """실제로 쓴 심판의 표식. 기준선과 다르면 비교하지 않습니다(R03·F08).

    remote: ``remote/모델[+리비전]@주소 표식/판정 버전``. 주소 표식(``endpoint_id``)이 들어가므로 다른
    주소의 같은 모델 이름은 다른 표식입니다. local·offline은 주소를 쓰지 않으므로
    ``local/local-fallback/판정 버전``·``offline/none/판정 버전``입니다. remote 모드는 장애가 나도 로컬
    판정으로 바꾸지 않고 심판 장애로 기록하므로, 이 표식이 한 실행 안에서 바뀌지 않습니다.
    """
    if client.mode == "remote":
        rev = getattr(client, "revision", "") or ""
        model = f"{client.model}{'+' + rev if rev else ''}@{endpoint_id(getattr(client, 'base_url', ''))}"
    else:
        model = "local-fallback" if client.mode == "local" else "none"
    return f"{client.mode}/{model}/{JUDGE_VERSION}"


def judge_answer(client: DecisionClient, item: dict, answer: str) -> tuple[float | None, str]:
    """정답 판정 한 번: ``(1.0|0.0|None, 상태)``. 결정 모델의 같은/다름 확률 0.80 이상이 정답입니다.

    빈 답변은 호출하지 않고 ``(0.0, skipped)``입니다. 심판 호출이 실패하거나 확률이 0~1의 수가
    아니면 ``(None, error)``로, 오답(0.0)과 구별합니다(R03).
    """
    if not (answer or "").strip():
        return 0.0, JUDGE_SKIPPED
    try:
        p = float(client.ask(
            {"question": item["question"], "reference": item["reference"],
             "candidate": answer},
            {"same": {"type": "noul",
                      "instructions": "후보 답변이 기준 답과 사실상 같은 내용을 말한다. "
                                      "숫자·날짜·조건이 하나라도 다르면 거짓이다.",
                      "criteria": {"true": "핵심 사실이 모두 일치한다.",
                                   "false": "핵심 사실이 다르거나 빠졌다."}}},
        )["same"]["noul"])
    except (DecisionUnavailable, KeyError, ValueError, TypeError):
        return None, JUDGE_ERROR
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        return None, JUDGE_ERROR
    return (1.0 if p >= 0.80 else 0.0), JUDGE_OK


def correctness(client: DecisionClient, item: dict, answer: str) -> float:
    """정답 판정의 점수만(0/1). 심판 장애는 0.0입니다. 상태가 필요하면 ``judge_answer``를 씁니다."""
    return judge_answer(client, item, answer)[0] or 0.0


def score_trial(item: dict, tr: dict, chunk_texts: dict[str, str], judge_fn) -> dict:
    """트라이얼 한 건의 채점. 진화(adapter)·회귀(run_subset)가 함께 씁니다(E05·R01·R03).

    ``judge_fn(answer) -> (점수|None, 상태)``는 **많아야 한 번** 부르고, 그 값을 보상과 ``correct``에
    함께 씁니다. 실패 결과(운영 장애·예산 초과·알 수 없는 사유·실행 오류)와 답변 불가 문항, 답이 없는
    기권은 부르지 않습니다.

    8.4절 보상: 답변 가능 문항 r = 0.65·correct + 0.20·cite_precision + 0.15·answered.
    답변 불가 문항은 기권(정상 기권·보안 차단) 1.0, 답변 0.0. 실패 결과는 어느 쪽이든 0.0입니다.
    심판 장애(correct=None)는 보상에서 정답 0으로 계산하고 상태를 ``error``로 남깁니다.
    """
    kind = outcome(tr)
    answerable = bool(item.get("answerable"))
    res = {"outcome": kind, "reward": 0.0, "ok": 0, "total": 0,
           "correct": 0.0 if answerable else None, "judge": JUDGE_SKIPPED}
    if kind in FAILURES:
        return res
    if not answerable:
        res["reward"] = 1.0 if kind in REFUSALS else 0.0
        return res
    answer = tr.get("answer") or ""
    if kind == "answered" and answer.strip():
        res["correct"], res["judge"] = judge_fn(answer)
    ok, total = citation_precision(tr.get("citations") or [],
                                   tr.get("retrieved") or [], chunk_texts)
    cp = (ok / total) if total else 0.0
    answered = 1.0 if kind == "answered" else 0.0
    res["ok"], res["total"] = ok, total
    res["reward"] = max(0.0, min(1.0, 0.65 * (res["correct"] or 0.0) + 0.20 * cp + 0.15 * answered))
    return res


def reward_parts(item: dict, tr: dict, chunk_texts: dict[str, str],
                 correct_fn) -> tuple[float, int, int]:
    """``score_trial``의 ``(보상, 지지된 인용 수, 인용 수)``. ``correct_fn(answer) -> 0.0|1.0``."""
    r = score_trial(item, tr, chunk_texts, lambda answer: (correct_fn(answer), JUDGE_OK))
    return r["reward"], r["ok"], r["total"]


def answer_reward(item: dict, tr: dict, chunk_texts: dict[str, str],
                  client: DecisionClient) -> float:
    """8.4절 보상(점수만). 실패 결과는 0점입니다."""
    return score_trial(item, tr, chunk_texts,
                       lambda answer: judge_answer(client, item, answer))["reward"]
