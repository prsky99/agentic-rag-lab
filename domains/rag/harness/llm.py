"""하네스의 모델 접점: 결정 모델(Jev)과 생성 LLM.

두 클라이언트 모두 같은 규칙을 따릅니다.

* 외부 엔드포인트가 설정되면 그쪽으로 요청합니다.
* 없으면 **로컬 대체 모드**로 동작합니다. 어휘 겹침과 패턴 규칙만 쓰는 결정적
  구현이며, 1~12장의 배선을 클러스터·모델 없이 확인하기 위한 것입니다.
  로컬 대체 모드의 판정은 정확도가 아니라 배선 검증이 목적입니다.
"""
from __future__ import annotations

import contextvars
import http.client
import json
import math
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager

TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")
STOP = {"이", "그", "저", "것", "수", "있다", "없다", "하다", "되다", "및", "그리고",
        "에서", "으로", "에게", "부터", "까지", "보다", "the", "a", "an", "of", "to",
        "is", "are", "and", "or", "in", "on", "for", "with", "할", "한", "된", "하는"}


class DecisionUnavailable(RuntimeError):
    """결정 모델 응답 없음. 게이트가 자기 정책대로 처리합니다."""


class LLMUnavailable(RuntimeError):
    """생성 모델 응답 없음. 하네스는 추출형 대체 답변으로 전환합니다."""


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(text or "") if len(t) > 1 and t not in STOP}


def token_match(t: str, w: str) -> bool:
    """한국어 조사·표기 차이를 흡수한 어휘 매칭."""
    if t == w:
        return True
    if len(t) >= 2 and (t in w or w in t):
        return True
    return len(t) >= 2 and len(w) >= 2 and t[:2] == w[:2]


def overlap(query: str, text: str) -> float:
    """질문 어휘가 텍스트에 얼마나 있는가. 한국어 조사 차이를 흡수합니다."""
    q = tokens(query)
    if not q:
        return 0.0
    d = tokens(text)
    hit = 0
    for t in q:
        if any(token_match(t, w) for w in d):
            hit += 1
    return hit / len(q)


def weighted_overlap(query: str, text: str, idf) -> float:
    """희소어에 가중치를 준 겹침. "환불"보다 "검토"가 더 큰 신호입니다."""
    q = tokens(query)
    if not q:
        return 0.0
    d = tokens(text)
    num = den = 0.0
    for t in q:
        w = max(0.01, float(idf(t)))
        den += w
        if any(token_match(t, x) for x in d):
            num += w
    return num / den if den else 0.0


DIGITS = re.compile(r"\d[\d,]*")


def digits(text: str) -> set[str]:
    return {m.replace(",", "") for m in DIGITS.findall(text or "")}


def estimate_tokens(text: str) -> int:
    """한국어 기준 대략 2.5자/토큰. 과금 계산이 아니라 예산 배분용입니다."""
    return max(1, int(len(text or "") / 2.5))


# 요청 한 건의 호출 범위. 파이프라인이 단계마다 남은 시간(timeout_s)을 넣고, 클라이언트는
# 실제 제공자 사용량(usage)을 여기에 적습니다. 제공자가 사용량을 주지 않았거나 형식이
# 틀렸으면 그 사유(usage_issue)를 적습니다. 요청마다 작업 스레드·컨텍스트가 다르므로
# 동시 요청끼리 값이 섞이지 않습니다.
_CALL_SCOPE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "harness_call_scope", default=None)


@contextmanager
def call_scope(timeout_s: float | None = None):
    """호출 한 번의 범위를 엽니다. 나올 때 ``scope["usage"]``에 제공자 사용량이 남습니다."""
    scope = {"timeout_s": timeout_s, "usage": None, "usage_issue": ""}
    token = _CALL_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CALL_SCOPE.reset(token)


def scoped_timeout(default: float) -> float:
    """남은 시간이 기본 타임아웃보다 짧으면 남은 시간을 씁니다(최소 0.001초)."""
    scope = _CALL_SCOPE.get()
    if scope and scope.get("timeout_s") is not None:
        return max(0.001, min(default, float(scope["timeout_s"])))
    return default


def _report_usage(usage: dict | None, issue: str = "") -> None:
    scope = _CALL_SCOPE.get()
    if scope is None:
        return
    if usage:
        scope["usage"] = dict(usage)
    elif issue:
        scope["usage_issue"] = issue


MAX_TOKENS = 10 ** 9        # 호출 한 번의 토큰 수로 믿을 수 있는 상한. 넘으면 invalid입니다.


def _num(v) -> bool:
    """유한한 수인가. 정수와 실수를 따로 검사합니다.

    JSON의 ``10**400`` 같은 큰 정수는 float로 바꿀 수 없어 ``math.isfinite``가
    ``OverflowError``를 냅니다. 정수는 그 자체로 유한하므로 isfinite에 넘기지 않습니다(RT11).
    """
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return isinstance(v, float) and math.isfinite(v)


def _count(v) -> int | None:
    """토큰 수로 쓸 수 있는 값(0 이상 MAX_TOKENS 이하의 정수)만 받습니다. 아니면 None입니다."""
    if not _num(v):
        return None
    if isinstance(v, float):
        if not v.is_integer():
            return None
        v = int(v)
    return v if 0 <= v <= MAX_TOKENS else None


def parse_usage(u, known: object = None) -> tuple[dict | None, str]:
    """제공자 응답의 usage를 검사해 ``(사용량, 사유)``를 돌려줍니다(RT06·RT11).

    사용량은 세 상태로 다룹니다.

    * 실측(provider): ``input_tokens``·``output_tokens``(또는 ``prompt_tokens``·
      ``completion_tokens``)가 모두 0 이상의 정수이고 둘 다 0은 아닌 경우입니다.
    * 추정(estimate): 원격 호출이 없는 로컬 연산을 글자 수로 어림한 값입니다(이 함수 밖).
    * 미측정(unknown): 요청은 보냈지만 믿을 수 있는 사용량이 없는 경우입니다. 이때 이 함수는
      ``None``과 사유를 돌려줍니다. 사유는 ``missing``(usage 없음), ``invalid``(dict가 아니거나
      값이 음수·비숫자·객체·상한 ``MAX_TOKENS`` 초과), ``not_counted``(응답의 ``usage_known``이 false이거나 0/0)입니다.

    형식이 틀려도 예외를 발생시키지 않습니다. 판단 값이 정상이면 판정은 그대로 쓰고,
    그 호출의 사용량만 미측정으로 남깁니다.
    """
    if u is None:
        return None, "missing"
    if not isinstance(u, dict):
        return None, "invalid"
    if not any(k in u for k in ("input_tokens", "prompt_tokens",
                                "output_tokens", "completion_tokens")):
        return None, "missing"
    i = _count(u.get("input_tokens", u.get("prompt_tokens")))
    o = _count(u.get("output_tokens", u.get("completion_tokens")))
    if i is None or o is None:
        return None, "invalid"
    if known is False or u.get("usage_known") is False or (i == 0 and o == 0):
        # 자체 결정 헤드는 토큰을 세지 않고 0/0을 돌려줍니다. 실측으로 받지 않습니다.
        return None, "not_counted"
    return {"input_tokens": i, "output_tokens": o}, ""


def validate_answers(questions: dict, answers) -> dict:
    """결정 모델 응답의 모양과 범위를 검사합니다.

    HTTP 200이어도 질문별 판단 값이 빠졌거나 범위를 벗어나면 ``DecisionUnavailable``을
    발생시킵니다. 게이트는 이 예외를 자기 정책(대체 판정·기권)으로 처리하므로,
    ``KeyError``가 요청 전체를 깨뜨리지 않습니다.
    """
    if not isinstance(answers, dict):
        raise DecisionUnavailable("invalid response: answers is not an object")
    for qid, q in questions.items():
        a = answers.get(qid)
        if not isinstance(a, dict):
            raise DecisionUnavailable(f"invalid response: missing answer {qid}")
        kind = (q or {}).get("type", "choice")
        if kind == "noul":
            v = a.get("noul")
            if not _num(v) or not 0.0 <= v <= 1.0:
                raise DecisionUnavailable(f"invalid response: {qid}.noul={v!r}")
        elif kind == "score":
            v = a.get("score")
            crit = (q or {}).get("criteria")
            top = (len(crit) - 1) if isinstance(crit, list) and crit else 4
            if not _num(v) or not 0.0 <= v <= top:
                raise DecisionUnavailable(f"invalid response: {qid}.score={v!r}")
        else:
            options = list(((q or {}).get("criteria") or {}).keys())
            choice = a.get("choice")
            probs = a.get("probabilities")
            if (not isinstance(choice, str) or (options and choice not in options)
                    or not isinstance(probs, dict)):
                raise DecisionUnavailable(f"invalid response: {qid}.choice={choice!r}")
            if choice not in probs or not all(_num(x) and 0.0 <= x <= 1.0
                                               for x in probs.values()):
                raise DecisionUnavailable(f"invalid response: {qid}.probabilities")
    return answers


DEADLINE_HEADER = "x-deadline-ms"      # 상위가 하위 서비스에 넘기는 남은 시간(ms)


def _abort(sock, fired: threading.Event) -> None:
    fired.set()
    for close in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
        try:
            close()
        except OSError:
            pass


def _trace_headers() -> dict:
    """현재 추적 문맥(traceparent 등). OpenTelemetry가 없거나 꺼져 있으면 빈 값입니다.

    이 모듈은 ``urllib.request.urlopen``을 거치지 않으므로, 서비스 공통 계측의 urlopen
    전파에 기대지 않고 여기서 직접 붙입니다.
    """
    try:
        from opentelemetry import propagate
    except ImportError:
        return {}
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    """JSON POST 한 번. ``timeout``은 요청 전체(연결·전송·헤더·본문 수신)의 상한입니다.

    소켓 타임아웃은 수신 한 번마다 다시 시작하므로, 본문을 조금씩 보내는 서버는 그 값만으로
    막지 못합니다. 그래서 연결 뒤에 감시 타이머를 걸고, 시간이 다 되면 소켓을 닫아 수신을
    끝냅니다. 시간이 넘으면 ``TimeoutError``를 발생시키고, 이쪽 프로세스에는 뒤에서 계속
    도는 작업이 남지 않습니다.

    이 상한이 덮지 못하는 범위는 두 가지입니다. 이름 풀이(DNS)는 운영 체제 호출이라 연결
    타임아웃과 따로 더 걸릴 수 있습니다. 그리고 상대 서버가 이미 시작한 처리는 이쪽이 소켓을
    닫아도 서버 안에서 계속될 수 있습니다. 그래서 생성 서비스에는 남은 시간을
    ``x-deadline-ms`` 헤더로 함께 보냅니다.
    """
    end = time.monotonic() + max(0.001, float(timeout))
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError(f"unsupported url: {url}")
    https = u.scheme == "https"
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    port = u.port or (443 if https else 80)
    proxy = "" if urllib.request.proxy_bypass(u.hostname) \
        else urllib.request.getproxies().get(u.scheme, "")
    cls = http.client.HTTPSConnection if https else http.client.HTTPConnection
    left = max(0.001, end - time.monotonic())
    if proxy:                                       # 환경 변수의 프록시 설정을 따릅니다
        pu = urllib.parse.urlsplit(proxy if "://" in proxy else "http://" + proxy)
        conn = cls(pu.hostname, pu.port or 80, timeout=left)
        if https:
            conn.set_tunnel(u.hostname, port)
        else:
            path = url
    else:
        conn = cls(u.hostname, port, timeout=left)
    fired = threading.Event()
    timer = None
    try:
        try:
            conn.connect()
            timer = threading.Timer(max(0.0, end - time.monotonic()), _abort,
                                    (conn.sock, fired))
            timer.daemon = True
            timer.start()
            conn.request("POST", path, body=data,
                         headers={"content-type": "application/json",
                                  **_trace_headers(), **headers})
            resp = conn.getresponse()
            raw = resp.read()
        except (OSError, http.client.HTTPException) as exc:
            if fired.is_set() or time.monotonic() >= end:
                raise TimeoutError(f"request deadline exceeded: {url}") from exc
            if isinstance(exc, OSError):
                raise
            raise urllib.error.URLError(repr(exc)) from exc
        if fired.is_set() or time.monotonic() >= end:
            # 길이 표시 없이 닫히는 응답은 잘린 본문이 정상처럼 보입니다. 시간으로 다시 가립니다.
            raise TimeoutError(f"request deadline exceeded: {url}")
        if resp.status >= 300:
            raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
        return json.loads(raw.decode("utf-8"))
    finally:
        if timer is not None:
            timer.cancel()
        conn.close()


WEB_CUES = ("최신", "오늘", "현재", "뉴스", "주가", "환율", "날씨", "공시", "실시간")
CHITCHAT = ("안녕", "고마워", "반가", "누구야", "뭐 해", "잘 지내")
INJECTION = (
    r"(?i)ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"이전\s*지시를?\s*무시",
    r"시스템\s*프롬프트[^.。\n]{0,20}(출력|알려|보여|공개)",
    r"(api|비밀)\s*키[^.。\n]{0,20}(출력|알려|보여|공개)",
    r"역할을?\s*바꿔",
    r"you\s+are\s+now",
    r"개발자\s*모드",
)
HUMAN_KEYWORDS = ("환불", "법", "계정", "삭제", "권한", "승인", "보상")


def looks_like_injection(text: str) -> bool:
    return any(re.search(p, text or "") for p in INJECTION)


class DecisionClient:
    """``/v1/systemone`` 클라이언트 + 로컬 대체 판정."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 1.5,
                 mode: str = "auto") -> None:
        self.base_url = (base_url if base_url is not None
                         else os.environ.get("DECISION_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("DECISION_MODEL", "jev-1.13.0")
        self.api_key = api_key or os.environ.get("DECISION_API_KEY", "")
        self.timeout = timeout
        if mode != "auto":
            self.mode = mode
        elif os.environ.get("DECISION_MODE") in ("local", "remote", "offline"):
            self.mode = os.environ["DECISION_MODE"]
        else:
            self.mode = "remote" if self.base_url else "local"
        # 짧은 회로 차단: 연속 실패가 쌓이면 잠시 호출 없이 바로 DecisionUnavailable을 발생시킵니다.
        # 결정 모델이 중단됐을 때 게이트마다 타임아웃(1.5초)을 반복해서 기다리는 일을 막습니다.
        self.fail_threshold, self.reset_after = 2, 10.0
        self._fails, self._opened_at = 0, 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def ask(self, state, questions: dict) -> dict:
        if self.mode == "offline":
            raise DecisionUnavailable("DECISION_MODE=offline")
        if self.mode == "remote":
            return self._ask_remote(state, questions)
        return self._ask_local(state, questions)

    def _ask_remote(self, state, questions: dict) -> dict:
        with self._lock:
            if (self._fails >= self.fail_threshold
                    and time.monotonic() - self._opened_at < self.reset_after):
                raise DecisionUnavailable("circuit open")
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            body = _post_json(self.base_url + "/v1/systemone",
                              {"model": self.model, "state": state, "questions": questions},
                              headers, scoped_timeout(self.timeout))
            if not isinstance(body, dict) or "answers" not in body:
                raise DecisionUnavailable("response has no answers")
            answers = validate_answers(questions, body["answers"])
            self._record(True)
            # 사용량 형식이 틀려도 판정은 그대로 씁니다. 그 호출만 미측정으로 남깁니다(RT11).
            _report_usage(*parse_usage(body.get("usage"), body.get("usage_known")))
            return answers
        except DecisionUnavailable:
            self._record(False)
            raise
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError,
                TypeError) as exc:
            self._record(False)
            raise DecisionUnavailable(str(exc)) from exc

    def _record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._fails = 0
                return
            self._fails += 1
            if self._fails >= self.fail_threshold:      # 열림(또는 반열림 시도 실패 → 다시 열림)
                self._opened_at = time.monotonic()

    # --- 로컬 대체 판정 ------------------------------------------------
    def _ask_local(self, state: dict, questions: dict) -> dict:
        out = {}
        for qid, q in questions.items():
            kind = q.get("type", "choice")
            if kind == "choice":
                options = list((q.get("criteria") or {"answer": ""}).keys())
                choice = self._choose(qid, state, options)
                share = round(0.12 / max(1, len(options) - 1), 4)
                probs = {o: (0.88 if o == choice else share) for o in options}
                out[qid] = {"type": "choice", "choice": choice,
                            "probabilities": probs, "confidence": probs[choice]}
            elif kind == "noul":
                out[qid] = {"type": "noul", "noul": round(self._noul(qid, state), 4)}
            else:
                out[qid] = {"type": "score", "score": self._score(qid, state)}
        return out

    def _choose(self, qid: str, state: dict, options: list[str]) -> str:
        question = str(state.get("question") or "")
        if qid == "path":
            if any(c in question for c in CHITCHAT) or len(question.strip()) <= 4:
                pick = "no_retrieval"
            elif any(c in question for c in WEB_CUES):
                pick = "web"
            else:
                pick = "local_docs"
        elif qid == "next":
            p_grounded = float(state.get("p_grounded") or 0.0)
            severity = float(state.get("severity") or 0.0)
            if p_grounded < 0.5 and any(k in question for k in HUMAN_KEYWORDS):
                pick = "human"
            elif severity >= 2.0:
                pick = "bigger_model"
            elif p_grounded >= 0.7:
                pick = "answer"
            else:
                pick = "abstain"
        elif qid == "strategy":
            if any(c in question for c in ("각각", "둘 다", "그리고", "비교")):
                pick = "decompose"
            elif any(c in question for c in ("최근", "지난", "이번", "올해", "기간")):
                pick = "constrain"
            elif any(c in question for c in ("리밋", "레이트", "반품", "환급")):
                pick = "normalize"
            else:
                pick = "expand"
        else:
            pick = options[0]
        return pick if pick in options else options[0]

    def _noul(self, qid: str, state: dict) -> float:
        question = str(state.get("question") or "")
        passages = state.get("passages") or []

        if qid.startswith("inj_") or qid == "override":
            texts = [str(p.get("text", "")) for p in passages] or [question]
            idx = None
            if qid.startswith("inj_"):
                try:
                    idx = int(qid.split("_", 1)[1])
                except (IndexError, ValueError):
                    idx = None
            checked = [texts[idx]] if idx is not None and idx < len(texts) else texts
            return 0.92 if any(looks_like_injection(t) for t in checked) else 0.02

        if qid in ("enough", "answerable"):
            if not passages:
                return 0.02
            text = " ".join(str(p.get("text", "")) for p in passages)
            cov = overlap(question, text)
            # 질문의 숫자·조건이 근거에 있으면 가산
            nums = digits(question)
            if nums and nums <= digits(text):
                cov = min(1.0, cov + 0.2)
            return max(0.02, min(0.96, 0.10 + 1.45 * cov))

        if qid == "grounded":
            answer = str(state.get("answer") or "")
            if not answer:
                return 0.0
            body = re.sub(r"\[\d+\]", "", answer)
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
            if not sentences:
                return 0.0
            span_list = [str(c.get("span", "")) for c in (state.get("citations") or [])]
            supported = 0.0
            for s in sentences:
                best = max((overlap(s, sp) for sp in span_list), default=0.0)
                if best >= 0.5:
                    supported += 1.0
                elif best >= 0.3:
                    supported += 0.5
            has_marker = bool(re.search(r"\[\d+\]", answer))
            p = supported / len(sentences)
            if has_marker and supported:
                p = min(1.0, p + 0.1)
            return round(min(0.98, 0.05 + 0.95 * p), 4)

        if qid == "needs_fresh":
            return 0.9 if any(c in question for c in WEB_CUES) else 0.05
        if qid == "multi_hop":
            return 0.72 if any(c in question for c in ("각각", "둘 다", "그리고", "비교")) else 0.12
        if qid in ("same", "correct"):
            ref = str(state.get("reference") or "")
            cand = str(state.get("candidate") or state.get("answer") or "")
            if not ref:
                return 0.0
            p = overlap(ref, cand)
            r = overlap(cand, ref)
            f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
            if digits(ref) - digits(cand):
                f1 *= 0.6                       # 숫자가 빠지면 오답으로 봅니다
            return round(min(0.97, 0.10 + 1.7 * f1), 4)
        return 0.5

    def _score(self, qid: str, state: dict) -> float:
        if qid == "severity":
            p = self._noul("grounded", state)
            return round(min(4.0, (1.0 - p) * 4.5), 2)
        text = " ".join(str(p.get("text", "")) for p in (state.get("passages") or []))
        return round(4 * min(1.0, overlap(str(state.get("question") or ""), text)), 2)


def _numbers(text: str) -> set[str]:
    return digits(text)


class LLMClient:
    """생성 LLM 클라이언트. 연결 대상이 두 종류이며 요청·응답 계약이 다릅니다.

    * ``provider="openai"``: 외부 모델 제공자. OpenAI 호환 ``POST {base}/chat/completions``.
      주소는 ``base_url`` 인자 또는 ``OPENAI_BASE_URL``, 키는 ``OPENAI_API_KEY``(없으면
      ``GENERATOR_API_KEY``)입니다. 키가 없으면 auto 모드에서 추출형으로 내려갑니다.
    * ``provider="generator"``: 동봉 생성 서비스(``services/generator``).
      ``POST {GENERATOR_URL}/generate``에 ``{system, user, temperature, max_tokens}``를 보내고
      ``{answer, usage, error}``를 받습니다. 키는 생성 서비스가 들고 있으므로 필요 없습니다.

    ``provider``를 지정하지 않으면 ``base_url``/``OPENAI_BASE_URL``이 있을 때 openai,
    ``GENERATOR_URL``만 있을 때 generator를 고릅니다.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 45.0,
                 mode: str = "auto", provider: str | None = None) -> None:
        openai_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        generator_url = os.environ.get("GENERATOR_URL", "")
        if provider is None:
            provider = "openai" if (openai_url or not generator_url) else "generator"
        if provider not in ("openai", "generator"):
            raise ValueError(f"unknown provider: {provider}")
        self.provider = provider
        self.base_url = (openai_url if provider == "openai"
                         else (base_url or generator_url)).rstrip("/")
        self.model = model or os.environ.get("GENERATOR_MODEL", "gpt-6-luna")
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY")
                        or os.environ.get("GENERATOR_API_KEY", ""))
        self.timeout = timeout
        if mode == "extractive":
            self.mode = "extractive"
        elif mode == "remote":
            self.mode = "remote"
        elif provider == "generator":
            self.mode = "remote" if self.base_url else "extractive"
        else:
            self.mode = "remote" if (self.base_url and self.api_key) else "extractive"

    def complete(self, system: str, user: str, temperature: float = 0.0,
                 max_tokens: int = 700) -> str:
        text, _ = self.complete_with_usage(system, user, temperature, max_tokens)
        return text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0,
                            max_tokens: int = 700) -> tuple[str, dict]:
        """생성 결과와 사용량을 돌려줍니다.

        ``usage["source"]``는 provider(실측) 또는 unknown(미측정)입니다. 미측정일 때
        ``input_tokens``·``output_tokens``는 글자 수 추정치이고, 사유는 ``usage["issue"]``에
        있습니다. 장부(``CallLedger``)는 미측정 호출을 합계에 넣지 않고 따로 셉니다.
        """
        if self.mode == "extractive":
            raise LLMUnavailable("extractive mode (no generation endpoint)")
        if not self.base_url:
            raise LLMUnavailable("no generation endpoint")
        timeout = scoped_timeout(self.timeout)
        try:
            if self.provider == "generator":
                body = _post_json(self.base_url + "/generate",
                                  {"system": system, "user": user,
                                   "temperature": temperature, "max_tokens": max_tokens},
                                  # 생성 서비스가 제공자 호출 시간을 남은 시간으로 줄입니다.
                                  {DEADLINE_HEADER: str(int(timeout * 1000))}, timeout)
                if (not isinstance(body, dict) or body.get("error")
                        or not isinstance(body.get("answer"), str) or not body["answer"]):
                    err = body.get("error") if isinstance(body, dict) else "invalid body"
                    raise LLMUnavailable(f"generator error: {err}")
                text = body["answer"]
                u = body.get("usage")
                # 생성 서비스가 source=provider로 표시한 값만 실측으로 받습니다.
                counted = not isinstance(u, dict) or u.get("source") == "provider"
                parsed, issue = parse_usage(u, None if counted else False)
            else:
                body = _post_json(
                    self.base_url + "/chat/completions",
                    {"model": self.model, "temperature": temperature,
                     "max_tokens": max_tokens,
                     "messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": user}]},
                    {"authorization": f"Bearer {self.api_key}"}, timeout)
                text = str(body["choices"][0]["message"]["content"])
                parsed, issue = parse_usage(body.get("usage"))
        except LLMUnavailable:
            raise
        except Exception as exc:                       # noqa: BLE001
            raise LLMUnavailable(str(exc)) from exc
        if parsed:
            usage = {**parsed, "source": "provider"}
        else:
            usage = {"input_tokens": estimate_tokens(system + user),
                     "output_tokens": estimate_tokens(text), "source": "unknown",
                     "issue": issue}
        _report_usage(parsed, issue)
        return text, usage
