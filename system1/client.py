"""결정 모델 클라이언트 (1.4절, 4.4절).

결정 모델은 ``/v1/systemone``을 노출한다고 가정합니다. 게이트는 이 클라이언트만
알고, 모델의 정체(OpenJev, Jev, vLLM 서빙)는 모릅니다. 4장에서 재시도 1회와
서킷 브레이커를 더해 운영 버전이 되었습니다.
"""
from __future__ import annotations

import logging
import math
import random
import time

import httpx

log = logging.getLogger("decision")


class DecisionUnavailable(RuntimeError):
    """결정 모델 응답 없음. 게이트는 이 예외를 자기 정책대로 처리합니다."""


def _num(v) -> bool:
    # 큰 JSON 정수(10**400 등)는 math.isfinite에서 OverflowError가 나므로 정수는 따로 봅니다(RT11).
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return isinstance(v, float) and math.isfinite(v)


def validate_answers(questions: dict, answers) -> dict:
    """HTTP 200 응답이라도 질문별 판단 값의 모양·범위를 검사합니다.

    빠졌거나 범위를 벗어나면 ``DecisionUnavailable``을 발생시켜 게이트가 자기 정책
    (대체 판정·기권)으로 처리하게 합니다. ``KeyError``로 요청이 깨지지 않습니다.
    """
    if not isinstance(answers, dict):
        raise DecisionUnavailable("invalid response: answers is not an object")
    for qid, q in questions.items():
        a = answers.get(qid)
        if not isinstance(a, dict):
            raise DecisionUnavailable(f"invalid response: missing answer {qid}")
        kind = (q or {}).get("type", "choice")
        if kind == "noul":
            if not _num(a.get("noul")) or not 0.0 <= a["noul"] <= 1.0:
                raise DecisionUnavailable(f"invalid response: {qid}.noul")
        elif kind == "score":
            crit = (q or {}).get("criteria")
            top = (len(crit) - 1) if isinstance(crit, list) and crit else 4
            if not _num(a.get("score")) or not 0.0 <= a["score"] <= top:
                raise DecisionUnavailable(f"invalid response: {qid}.score")
        else:
            options = list(((q or {}).get("criteria") or {}).keys())
            choice, probs = a.get("choice"), a.get("probabilities")
            if (not isinstance(choice, str) or (options and choice not in options)
                    or not isinstance(probs, dict) or choice not in probs
                    or not all(_num(x) and 0.0 <= x <= 1.0 for x in probs.values())):
                raise DecisionUnavailable(f"invalid response: {qid}.choice")
    return answers


class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, reset_after: float = 10.0):
        self.fail_threshold = fail_threshold
        self.reset_after = reset_after
        self.fails = 0
        self.opened_at = 0.0

    def allow(self) -> bool:
        if self.fails < self.fail_threshold:
            return True
        if time.monotonic() - self.opened_at >= self.reset_after:
            self.fails = 0          # half-open: 한 번 시도해 봅니다
            return True
        return False

    def record(self, ok: bool) -> None:
        if ok:
            self.fails = 0
        else:
            self.fails += 1
            if self.fails == self.fail_threshold:
                self.opened_at = time.monotonic()
                log.warning("decision circuit opened")


class DecisionClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 1.5, retries: int = 1) -> None:
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._model = model
        self._retries = retries
        self._breaker = CircuitBreaker()
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                                  timeout=httpx.Timeout(timeout, connect=0.3))

    def ask(self, state, questions: dict) -> dict:
        if not self._breaker.allow():
            raise DecisionUnavailable("circuit open")
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                r = self._http.post("/v1/systemone",
                                    json={"model": self._model, "state": state,
                                          "questions": questions})
                if r.status_code in (400, 401, 422):
                    # 재시도해도 같은 결과입니다. 계약 오류는 즉시 예외로 올립니다.
                    self._breaker.record(False)
                    raise DecisionUnavailable(f"contract error {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                try:
                    body = r.json()
                    answers = validate_answers(questions, (body or {}).get("answers")
                                               if isinstance(body, dict) else None)
                except (ValueError, DecisionUnavailable) as exc:
                    # 200이지만 계약을 어긴 응답입니다. 재시도해도 같으므로 즉시 올립니다.
                    self._breaker.record(False)
                    raise DecisionUnavailable(f"invalid response: {exc}") from exc
                self._breaker.record(True)
                return answers
            except httpx.HTTPError as e:
                last = e
                self._breaker.record(False)
                if attempt < self._retries:
                    time.sleep(0.05 + random.random() * 0.1)   # 지터
        raise DecisionUnavailable(str(last))

    def close(self) -> None:
        self._http.close()
