"""OTel 공용 초기화 시험 (10.1절).

* OTEL_SDK_DISABLED=true면 Provider를 켜지 않습니다.
* 미들웨어가 들어온 traceparent를 이어받아 서버 스팬을 엽니다.
* 나가는 urllib 요청에 같은 추적의 traceparent가 붙습니다.
opentelemetry-sdk·fastapi가 없으면 건너뜁니다(requirements.txt에 고정).
"""
from __future__ import annotations

import urllib.request

import pytest

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("fastapi")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter)

from services.common import otel  # noqa: E402

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT = f"00-{TRACE_ID}-00f067aa0ba902b7-01"


@pytest.fixture(scope="module")
def exporter():
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    trace.set_tracer_provider(tp)
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        pytest.skip("다른 TracerProvider가 이미 설정되어 있습니다")
    return exp


def test_disabled_flag_turns_off(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    monkeypatch.setattr(otel, "_initialized", False)
    assert otel.init_otel("t") is False


def test_middleware_continues_trace_and_propagates(exporter, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    seen = {}

    def fake_urlopen(req, *a, **kw):
        seen["traceparent"] = req.get_header("Traceparent")
        return None

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(otel, "_patched", False)
    otel._patch_clients()                    # fake_urlopen 위에 전파 래퍼를 씌웁니다

    app = FastAPI()
    otel.instrument_app(app)

    @app.get("/v1/ping")
    def ping():
        urllib.request.urlopen(urllib.request.Request("http://decision:8000/v1/version"))
        return {"trace": otel.current_trace_id()}

    exporter.clear()
    r = TestClient(app).get("/v1/ping", headers={"traceparent": PARENT})
    assert r.status_code == 200
    assert r.json()["trace"] == TRACE_ID
    assert seen["traceparent"] and seen["traceparent"].split("-")[1] == TRACE_ID
    server = [s for s in exporter.get_finished_spans() if s.kind == trace.SpanKind.SERVER]
    assert server and format(server[0].context.trace_id, "032x") == TRACE_ID
    assert server[0].attributes["http.response.status_code"] == 200
