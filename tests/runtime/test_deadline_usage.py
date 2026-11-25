"""제한 시간·사용량 회귀 시험(2차 검토 RT04·RT06·RT11).

검토에서 나온 조건을 그대로 재현합니다. 본문을 조금씩 보내는 로컬 HTTP 서버, 0ms 예산,
토큰을 세지 않는 결정 헤드의 0/0, 형식이 틀린 usage를 씁니다. 모델·클러스터는 필요 없습니다.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness import llm
from harness.budget import BudgetExceeded, CallLedger
from harness.pipeline import Deadline
from harness.retrieve import HttpRetriever, RetrievalUnavailable

REFUND_Q = "환불은 결제일로부터 며칠 이내에 요청할 수 있나요?"
ANSWER = "환불은 결제일로부터 14일 이내에 요청할 수 있습니다. [1]"


class _Handler(BaseHTTPRequestHandler):
    """경로별로 결정·생성·검색 응답을 흉내 냅니다. ``server.drip``에 든 경로는 본문을 조금씩 보냅니다."""

    def log_message(self, *a):                       # 시험 출력을 조용하게 둡니다
        pass

    def do_POST(self):
        srv = self.server
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        srv.seen.append((self.path, dict(self.headers)))
        if self.path == "/v1/systemone":
            answers = llm.DecisionClient(mode="local").ask(req["state"], req["questions"])
            body = {"answers": answers, **srv.decision_extra}
            slow = "grounded" in req["questions"] and "g3" in srv.drip
        elif self.path == "/generate":
            body = {"answer": ANSWER, "error": None,
                    "usage": {"input_tokens": 50, "output_tokens": 9, "source": "provider"}}
            slow = "generate" in srv.drip
        elif self.path == "/chat/completions":
            body = {"choices": [{"message": {"content": ANSWER}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 9}}
            slow = "provider" in srv.drip
        else:                                        # /search
            body = {"results": [], "pad": "x" * 400}
            slow = "search" in srv.drip
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        try:
            if not slow:
                self.wfile.write(raw)
                return
            for i in range(0, len(raw), 8):          # 8바이트씩 30ms 간격
                self.wfile.write(raw[i:i + 8])
                self.wfile.flush()
                time.sleep(0.03)
        except OSError:                              # 클라이언트가 시간 초과로 끊음
            pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    srv.drip, srv.seen, srv.decision_extra = set(), [], {}
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02},
                     daemon=True).start()
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv
    srv.shutdown()
    srv.server_close()


def remote_decision(server) -> llm.DecisionClient:
    return llm.DecisionClient(base_url=server.url, mode="remote")


# ---------------------------------------------------------------------------
# RT04 응답을 받는 동안에도 전체 제한 시간 적용
# ---------------------------------------------------------------------------
def test_rt04_post_json_total_timeout_covers_dripped_body(server):
    server.drip.add("search")
    t = time.monotonic()
    with pytest.raises(TimeoutError):
        llm._post_json(server.url + "/search", {"query": "q"}, {}, 0.06)
    # 조각마다 타임아웃이 다시 시작하면 본문이 다 올 때까지(수백 ms) 기다립니다.
    assert (time.monotonic() - t) < 0.3


def test_rt04_dripped_g3_response_does_not_answer_within_60ms(pipeline, server):
    """검토자의 재현 조건: 60ms 예산, 마지막 호출(G3)의 본문이 조금씩 도착합니다."""
    server.drip.add("g3")
    pipeline.decision = remote_decision(server)
    t = time.monotonic()
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro", deadline_ms=60)
    elapsed_ms = (time.monotonic() - t) * 1000
    assert res["status"] == "abstained" and res["reason"] == "deadline_exceeded"
    assert res["answer"] == "" and elapsed_ms < 400


def test_rt04_dripped_generation_does_not_answer(pipeline, server):
    server.drip.add("generate")
    pipeline.llm = llm.LLMClient(provider="generator", base_url=server.url)
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro", deadline_ms=200)
    assert res["status"] == "abstained" and res["reason"] == "deadline_exceeded"
    gen = [c for c in res["usage"]["calls"] if c["stage"] == "generate"]
    assert gen and gen[0]["source"] == "unknown" and gen[0]["issue"] == "failed"


def test_rt04_expiry_checked_after_call_and_before_return(pipeline):
    """마지막 호출이 예산을 넘겨 끝나면, 판정이 통과여도 answered를 내지 않습니다."""

    class SlowG3:
        mode = "local"

        def __init__(self, inner):
            self.inner = inner

        def ask(self, state, questions):
            if "grounded" in questions:
                time.sleep(0.25)
            return self.inner.ask(state, questions)

    pipeline.decision = SlowG3(pipeline.decision)
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro", deadline_ms=150)
    assert res["status"] == "abstained" and res["reason"] == "deadline_exceeded"
    assert any(f.startswith("budget:deadline@") for f in res["flags"])


def test_rt04_final_exit_blocks_expired_answer(pipeline):
    from harness.pipeline import Evidence
    run = pipeline._new_run("pro", 0, Evidence(), None)
    res = pipeline._finish(run, "answered", "답 [1]", [], False)
    assert res["status"] == "abstained" and res["reason"] == "deadline_exceeded"


def test_rt04_zero_deadline_is_expired_not_default(pipeline):
    calls = []

    class Probe:
        mode = "local"

        def ask(self, state, questions):
            calls.append(1)
            raise AssertionError("만료된 요청은 호출하지 않습니다")

    pipeline.decision = Probe()
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro", deadline_ms=0)
    assert res["status"] == "abstained" and res["reason"] == "deadline_exceeded"
    assert res["policy"]["deadline_ms"] == 0 and not calls
    # None만 설정의 기본값을 씁니다.
    assert pipeline._new_run("pro", None, None, None).deadline.total_ms == 6000


def test_rt04_orchestrator_api_zero_deadline(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from services.orchestrator import main as orch
    r = TestClient(orch.app).post(
        "/v1/answer", json={"question": REFUND_Q, "tenant_id": "acme", "deadline_ms": 0},
        headers={"x-internal-token": orch.INTERNAL_TOKEN})
    assert r.status_code == 200
    assert r.json()["status"] == "abstained" and r.json()["reason"] == "deadline_exceeded"


# ---------------------------------------------------------------------------
# RT04 검색 시간 초과·실패는 500이 아니라 통제된 기권
# ---------------------------------------------------------------------------
def test_rt04_http_retriever_timeout_becomes_retrieval_unavailable(server):
    server.drip.add("search")
    with llm.call_scope(0.05):
        with pytest.raises(RetrievalUnavailable):
            HttpRetriever(server.url).search("q", "acme")


def test_rt04_search_timeout_is_controlled_abstention_in_api(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from services.orchestrator import main as orch

    class TimingOut:
        def search(self, question, tenant, top_k=40):
            raise TimeoutError("timed out")

    pipe = orch._pipeline()
    monkeypatch.setattr(pipe, "retriever", TimingOut())
    r = TestClient(orch.app).post(
        "/v1/answer", json={"question": REFUND_Q, "tenant_id": "acme"},
        headers={"x-internal-token": orch.INTERNAL_TOKEN})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "abstained" and body["reason"] == "retrieval_unavailable"
    assert "retrieval_unavailable" in body["flags"]


def test_rt04_search_failure_counts_as_error_in_evaluation(pipeline, server):
    """검색 실패로 생긴 기권은 평가에서 정상 기권이 아니라 실행 오류입니다(E05)."""
    server.shutdown()
    server.server_close()
    pipeline.retriever = HttpRetriever(server.url)
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro")
    assert res["status"] == "abstained" and res["reason"] == "retrieval_unavailable"
    assert res["error"] and res["error"].startswith("retrieval_unavailable")


def test_rt04_search_timeout_by_deadline_is_deadline_exceeded(pipeline, server):
    server.drip.add("search")
    pipeline.retriever = HttpRetriever(server.url)
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro", deadline_ms=80)
    assert res["status"] == "abstained" and res["reason"] == "deadline_exceeded"


# ---------------------------------------------------------------------------
# RT04 남은 시간을 생성 서비스의 제공자 호출까지 전달
# ---------------------------------------------------------------------------
def test_rt04_remaining_time_header_is_sent_to_generator(pipeline, server):
    pipeline.llm = llm.LLMClient(provider="generator", base_url=server.url)
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro", deadline_ms=3000)
    assert [c["source"] for c in res["usage"]["calls"] if c["stage"] == "generate"] == ["provider"]
    sent = [h for path, h in server.seen if path == "/generate"]
    assert sent and 0 < int(sent[0][llm.DEADLINE_HEADER]) <= 3000


def test_rt04_generator_service_limits_provider_call(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from services.generator import main as gen
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("GENERATOR_API_KEY", "k")
    timeouts = []

    def fake_post(url, payload, headers, timeout):
        timeouts.append(timeout)
        return {"choices": [{"message": {"content": "답"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

    monkeypatch.setattr(llm, "_post_json", fake_post)
    api = TestClient(gen.app)
    body = {"system": "s", "user": "u"}
    assert api.post("/generate", json=body).json()["answer"] == "답"
    assert api.post("/generate", json=body, headers={"x-deadline-ms": "250"}).json()["answer"]
    assert timeouts == [45.0, 0.25]              # 헤더가 없으면 기본값, 있으면 남은 시간
    out = api.post("/generate", json=body, headers={"x-deadline-ms": "0"}).json()
    assert out["answer"] == "" and out["error"] == "deadline_exceeded"
    assert out["usage"]["source"] == "unknown" and len(timeouts) == 2


def test_rt04_generator_service_stops_dripping_provider(monkeypatch, server):
    """생성 서비스가 받은 남은 시간이 제공자 응답 수신 전체에 걸립니다."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from services.generator import main as gen
    server.drip.add("provider")
    monkeypatch.setenv("OPENAI_BASE_URL", server.url)
    monkeypatch.setenv("GENERATOR_API_KEY", "k")
    t = time.monotonic()
    out = TestClient(gen.app).post("/generate", json={"system": "s", "user": "u"},
                                   headers={"x-deadline-ms": "60"}).json()
    assert out["answer"] == "" and out["error"]
    assert (time.monotonic() - t) < 0.4


def test_rt04_reference_orchestrator_checks_expiry():
    pytest.importorskip("httpx")
    from services.orchestrator.orchestrate import (OrchestrateRequest, Result, Route, guarded,
                                                   orchestrate)
    from services.orchestrator.deadline import Deadline as RefDeadline
    from services.orchestrator.policy_store import PolicySnapshot

    class Deps:
        def route(self, *a, **k):
            raise AssertionError("만료된 요청은 라우팅하지 않습니다")

    policy = PolicySnapshot.__new__(PolicySnapshot)
    object.__setattr__(policy, "version", "t")
    res = orchestrate(OrchestrateRequest("q", "acme", deadline_ms=0), Deps(), policy)
    assert res.status == "abstained" and res.reason == "deadline_exceeded"
    late = guarded(Result("answered", answer="답", policy_version="t"), RefDeadline.start(0))
    assert late.status == "abstained" and late.reason == "deadline_exceeded"
    assert Route("local_docs", "x").path == "local_docs"


def test_rt04_ledger_check_after_call():
    ledger = CallLedger(Deadline.start(0), 5, 2)
    with pytest.raises(BudgetExceeded) as exc:
        ledger.check_deadline("g3")
    assert exc.value.reason == "deadline"


# ---------------------------------------------------------------------------
# RT06 사용량 세 상태: 실측(provider)·추정(estimate)·미측정(unknown)
# ---------------------------------------------------------------------------
def test_rt06_local_run_is_all_estimate_and_known(pipeline):
    u = pipeline.answer_sync(REFUND_Q, "acme", "pro")["usage"]
    assert u["source"] == "estimate" and u["usage_known"] is True and u["unknown_calls"] == 0
    assert "total_is_minimum" not in u and "unknown_estimate_tokens" not in u
    assert all(set(c) == {"stage", "kind", "input_tokens", "output_tokens", "source"}
               for c in u["calls"])


@pytest.mark.parametrize("extra", [
    {"usage": {"input_tokens": 0, "output_tokens": 0}},                       # 헤드의 고정 0/0
    {"usage": {"input_tokens": 0, "output_tokens": 0}, "usage_known": False},  # 새 헤드 계약
    {"usage": {"input_tokens": 7, "output_tokens": 1}, "usage_known": False},
    {},                                                                       # usage 없음
])
def test_rt06_uncounted_decision_usage_is_unknown_not_provider(pipeline, server, extra):
    server.decision_extra = extra
    pipeline.decision = remote_decision(server)
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro")
    u = res["usage"]
    dec = [c for c in u["calls"] if c["kind"] == "decision"]
    assert res["status"] == "answered" and len(dec) == 4          # g1·screen·g2·g3
    assert all(c["source"] == "unknown" and c["input_tokens"] == 0 for c in dec)
    assert all(c["estimate_tokens"] > 0 and c["issue"] in ("not_counted", "missing") for c in dec)
    assert u["provider_tokens"] == 0 and u["gate_tokens"] == 0
    assert u["usage_known"] is False and u["unknown_calls"] == 4
    assert u["total_is_minimum"] is True
    assert u["unknown_estimate_tokens"] == sum(c["estimate_tokens"] for c in dec)


def test_rt06_counted_decision_usage_is_provider(pipeline, server):
    server.decision_extra = {"usage": {"input_tokens": 120, "output_tokens": 2}}
    pipeline.decision = remote_decision(server)
    u = pipeline.answer_sync(REFUND_Q, "acme", "pro")["usage"]
    dec = [c for c in u["calls"] if c["kind"] == "decision"]
    assert all(c["source"] == "provider" and c["input_tokens"] == 120 for c in dec)
    assert u["gate_tokens"] == 4 * 122 and u["unknown_calls"] == 0


def test_rt06_failed_generation_call_is_unknown(pipeline):
    class Failing:
        mode = "remote"

        def complete_with_usage(self, *a, **k):
            raise llm.LLMUnavailable("503")

    pipeline.llm = Failing()
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro")
    u = res["usage"]
    failed = [c for c in u["calls"] if c["kind"] == "llm"]
    assert res["status"] == "answered"                      # 추출형 대체 답변
    assert len(failed) == 1 and failed[0]["source"] == "unknown"
    assert failed[0]["issue"] == "failed" and failed[0]["estimate_tokens"] > 0
    assert u["usage_known"] is False and u["unknown_calls"] == 1 and u["source"] == "mixed"


def test_rt06_generation_without_provider_usage_is_unknown(monkeypatch):
    monkeypatch.setattr(llm, "_post_json",
                        lambda *a, **k: {"choices": [{"message": {"content": "답입니다"}}]})
    _, usage = llm.LLMClient(base_url="https://api.example/v1", api_key="k") \
        .complete_with_usage("s", "u")
    assert usage["source"] == "unknown" and usage["issue"] == "missing"
    # 생성 서비스가 source=provider로 표시하지 않은 값도 실측으로 받지 않습니다.
    monkeypatch.setattr(llm, "_post_json", lambda *a, **k: {
        "answer": "답", "usage": {"input_tokens": 3, "output_tokens": 1, "source": "estimate"}})
    _, usage = llm.LLMClient(provider="generator", base_url="http://generator:8080") \
        .complete_with_usage("s", "u")
    assert usage["source"] == "unknown" and usage["issue"] == "not_counted"


# 평가 비용 집계(test_rt06_evaluation_cost_fills_unknown_with_estimate)는 진화 루프의 어댑터를 쓰므로
# tests/meta/test_eval_cost.py로 옮겼습니다.


def test_rt06_decision_head_contract_marks_usage_unknown():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "decision_head" / "server.py") \
        .read_text(encoding="utf-8")
    assert '"usage_known": False' in src


# ---------------------------------------------------------------------------
# RT11 usage 형식 오류: TypeError 대신 미측정 + flags 표시
# ---------------------------------------------------------------------------
BAD_USAGE = [
    {"input_tokens": 10, "output_tokens": {"invalid": 1}},     # 검토자의 재현 값
    {"input_tokens": -1, "output_tokens": 3},
    {"input_tokens": "10", "output_tokens": 3},
    {"input_tokens": 10, "output_tokens": None},
    {"input_tokens": 1.5, "output_tokens": 3},
    {"input_tokens": True, "output_tokens": 3},
    [10, 3],
    "10/3",
    {"input_tokens": 10 ** 400, "output_tokens": 3},          # 3차 검토 재현 값(OverflowError)
    {"input_tokens": 10, "output_tokens": 1e300},             # 상한(MAX_TOKENS)을 넘는 실수
    {"input_tokens": 10, "output_tokens": float("inf")},
]


@pytest.mark.parametrize("usage", BAD_USAGE)
def test_rt11_parse_usage_never_raises(usage):
    assert llm.parse_usage(usage) == (None, "invalid")


def test_rt11_parse_usage_accepts_valid_shapes():
    assert llm.parse_usage({"input_tokens": 10, "output_tokens": 3}) == \
        ({"input_tokens": 10, "output_tokens": 3}, "")
    assert llm.parse_usage({"prompt_tokens": 10.0, "completion_tokens": 0})[0] == \
        {"input_tokens": 10, "output_tokens": 0}
    assert llm.parse_usage(None) == (None, "missing")
    assert llm.parse_usage({"input_tokens": 0, "output_tokens": 0}) == (None, "not_counted")


@pytest.mark.parametrize("usage", BAD_USAGE[:2] + BAD_USAGE[6:7])
def test_rt11_invalid_decision_usage_keeps_verdict_and_flags(pipeline, server, usage):
    server.decision_extra = {"usage": usage}
    pipeline.decision = remote_decision(server)
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro")          # TypeError가 나오면 안 됩니다
    assert res["status"] == "answered" and res["decisions"]["g3"]["ok"] is True
    assert "g1_fallback" not in res["flags"] and "g2_fallback" not in res["flags"]
    assert {"usage_invalid@g1", "usage_invalid@g2", "usage_invalid@g3"} <= set(res["flags"])
    dec = [c for c in res["usage"]["calls"] if c["kind"] == "decision"]
    assert all(c["source"] == "unknown" and c["issue"] == "invalid" for c in dec)
    assert res["usage"]["usage_known"] is False


def test_rt11_invalid_generation_usage_is_unknown(pipeline, monkeypatch):
    monkeypatch.setattr(llm, "_post_json", lambda *a, **k: {
        "choices": [{"message": {"content": ANSWER}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": {"invalid": 1}}})
    pipeline.llm = llm.LLMClient(base_url="https://api.example/v1", api_key="k")
    res = pipeline.answer_sync(REFUND_Q, "acme", "pro")
    gen = [c for c in res["usage"]["calls"] if c["stage"] == "generate"]
    assert res["error"] is None and gen[0]["source"] == "unknown"   # TypeError 없이 끝납니다
    assert gen[0]["issue"] == "invalid" and "usage_invalid@generate" in res["flags"]


def test_rt11_huge_integer_usage_keeps_answer_in_api(server, monkeypatch):
    """JSON의 10**400 사용량이 OverflowError → HTTP 500으로 번지지 않습니다(3차 검토)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from services.orchestrator import main as orch
    server.decision_extra = {"usage": {"input_tokens": 10 ** 400, "output_tokens": 3}}
    pipe = orch._pipeline()
    monkeypatch.setattr(pipe, "decision", remote_decision(server))
    r = TestClient(orch.app).post(
        "/v1/answer", json={"question": REFUND_Q, "tenant_id": "acme"},
        headers={"x-internal-token": orch.INTERNAL_TOKEN})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and "usage_invalid@g3" in body["flags"]


def test_rt11_huge_integer_decision_value_is_unavailable_not_overflow():
    q = {"ok": {"type": "noul"}}
    with pytest.raises(llm.DecisionUnavailable):
        llm.validate_answers(q, {"ok": {"noul": 10 ** 400}})
    pytest.importorskip("httpx")
    from system1.client import DecisionUnavailable, validate_answers
    with pytest.raises(DecisionUnavailable):
        validate_answers(q, {"ok": {"noul": 10 ** 400}})


def test_rt04_trace_context_is_propagated_without_urlopen(server):
    """urlopen을 거치지 않아도 추적 문맥(traceparent)이 나가는 요청에 붙습니다."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    tracer = TracerProvider().get_tracer("t")
    with trace.use_span(tracer.start_span("s"), end_on_exit=True):
        llm._post_json(server.url + "/search", {"query": "q"}, {}, 2.0)
    headers = {k.lower(): v for k, v in server.seen[-1][1].items()}
    assert headers.get("traceparent", "").startswith("00-")
