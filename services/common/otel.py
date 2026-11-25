"""OpenTelemetry 공용 초기화와 스팬 헬퍼 (2.5절, 10.1절).

서비스 코드는 ``stage()``와 ``current_trace_id()``만 씁니다. 나머지는 이 모듈이 맡습니다.

* ``init_otel()`` — TracerProvider·MeterProvider와 OTLP(HTTP) 내보내기를 한 번만 설정합니다.
  서비스 이름은 ``OTEL_SERVICE_NAME``(없으면 uvicorn 인자의 services.<이름>.main), 수집기 주소는 ``OTEL_EXPORTER_OTLP_ENDPOINT``
  (예: http://otel-collector:4318)에서 읽습니다.
* ``OtelASGIMiddleware`` — 들어오는 요청의 ``traceparent``를 꺼내 서버 스팬을 열고,
  요청 수·지연 지표(``http.server.request.duration``)를 기록합니다.
* 나가는 요청 — httpx(Client·AsyncClient)와 urllib.request.urlopen에 현재 추적 문맥
  (W3C traceparent)을 자동으로 넣습니다. 서비스·하네스 코드를 고치지 않아도
  gateway → orchestrator → retriever·decision이 한 추적으로 이어집니다.
* ``autoinstrument()`` — 위 세 가지를 켜고, 이후 만들어지는 FastAPI 앱마다 미들웨어를
  붙입니다. ``services/__init__.py``가 프로세스 시작 시 한 번 부릅니다.

``OTEL_SDK_DISABLED=true``이거나 OTel 패키지가 없거나 수집기 주소가 없으면 아무것도
켜지 않습니다(no-op). 이때도 ``stage()``는 그대로 동작합니다.
"""
from __future__ import annotations

import os
import re
import sys
import time
from contextlib import contextmanager

try:
    from opentelemetry import context as otel_context
    from opentelemetry import metrics, propagate, trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    tracer = trace.get_tracer("rag")
except Exception:                                   # noqa: BLE001
    trace = None                                    # OTel이 없으면 no-op
    tracer = None

LATENCY_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 15]   # p95 목표 3초에 맞춘 버킷(초)
_initialized = False
_patched = False
_instruments: dict = {}


def otel_disabled() -> bool:
    return os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() in {"1", "true", "yes"}


def _guess_service_name() -> str:
    """``uvicorn services.gateway.main:app`` → ``rag-gateway``. 못 찾으면 ``rag``."""
    for arg in sys.argv:
        m = re.match(r"services\.(\w+)\.main", arg)
        if m:
            return f"rag-{m.group(1)}"
    return "rag"


def init_otel(service_name: str | None = None) -> bool:
    """Provider와 Exporter를 설정합니다. 실제로 켰으면 True를 돌려줍니다."""
    global _initialized
    if _initialized:
        return True
    if trace is None or otel_disabled() or not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:                             # API만 있고 SDK·Exporter가 없음
        return False

    name = service_name or os.environ.get("OTEL_SERVICE_NAME") or _guess_service_name()
    resource = Resource.create({"service.name": name,
                                "service.namespace": "agentic-rag"})
    # 엔드포인트·프로토콜·헤더는 OTEL_EXPORTER_OTLP_* 환경 변수에서 Exporter가 직접 읽습니다.
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tp)
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(),
        export_interval_millis=int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "15000")))
    views = [View(instrument_name=n,
                  aggregation=ExplicitBucketHistogramAggregation(LATENCY_BUCKETS))
             for n in ("http.server.request.duration", "rag_request_duration_seconds")]
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader],
                                             views=views))
    _initialized = True
    return True


def _meter_instruments() -> dict:
    if not _instruments and trace is not None:
        m = metrics.get_meter("rag.http")
        # 버킷 경계는 init_otel()의 View가 정합니다(API 1.29에는 advisory 인자가 없습니다)
        _instruments["dur"] = m.create_histogram("http.server.request.duration", unit="s")
        _instruments["req"] = m.create_counter("http.server.requests")
    return _instruments


class OtelASGIMiddleware:
    """서버 스팬 + 요청 지표. 요청 처리 로직에는 관여하지 않습니다."""

    SKIP = {"/healthz", "/ready", "/metrics"}       # 프로브·스크레이프는 추적하지 않습니다

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if trace is None or scope.get("type") != "http" or scope.get("path") in self.SKIP:
            await self.app(scope, receive, send)
            return
        carrier = {k.decode("latin-1"): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        ctx = propagate.extract(carrier)
        route = scope.get("path", "")
        method = scope.get("method", "GET")
        status = {"code": 500}

        async def _send(message):
            if message.get("type") == "http.response.start":
                status["code"] = int(message.get("status", 500))
            await send(message)

        start = time.perf_counter()
        token = otel_context.attach(ctx)
        try:
            with tracer.start_as_current_span(f"{method} {route}", kind=SpanKind.SERVER) as sp:
                sp.set_attribute("http.request.method", method)
                sp.set_attribute("url.path", route)
                try:
                    await self.app(scope, receive, _send)
                finally:
                    sp.set_attribute("http.response.status_code", status["code"])
                    if status["code"] >= 500:
                        sp.set_status(Status(StatusCode.ERROR))
        finally:
            otel_context.detach(token)
            inst = _meter_instruments()
            attrs = {"http.request.method": method, "http.route": route,
                     "http.response.status_code": status["code"]}
            inst["dur"].record(time.perf_counter() - start, attrs)
            inst["req"].add(1, attrs)


def _inject(headers) -> None:
    """현재 문맥의 traceparent를 헤더에 넣습니다. 이미 있으면 그대로 둡니다."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    for k, v in carrier.items():
        if k not in headers:
            headers[k] = v


def _patch_clients() -> None:
    """httpx·urllib의 나가는 요청에 추적 문맥을 붙입니다(한 번만)."""
    global _patched
    if _patched or trace is None:
        return
    _patched = True
    try:
        import httpx

        orig_send, orig_asend = httpx.Client.send, httpx.AsyncClient.send

        def send(self, request, *a, **kw):
            _inject(request.headers)
            return orig_send(self, request, *a, **kw)

        async def asend(self, request, *a, **kw):
            _inject(request.headers)
            return await orig_asend(self, request, *a, **kw)

        httpx.Client.send, httpx.AsyncClient.send = send, asend
    except ImportError:
        pass

    import urllib.request

    orig_urlopen = urllib.request.urlopen

    def urlopen(url, *a, **kw):
        if isinstance(url, urllib.request.Request):
            carrier: dict[str, str] = {}
            propagate.inject(carrier)
            for k, v in carrier.items():
                if not url.has_header(k.capitalize()):
                    url.add_header(k, v)
        return orig_urlopen(url, *a, **kw)

    urllib.request.urlopen = urlopen


def instrument_app(app) -> None:
    """FastAPI(Starlette) 앱에 미들웨어를 붙입니다. 이미 붙어 있으면 건너뜁니다."""
    if getattr(app.state, "_otel_instrumented", False):
        return
    app.add_middleware(OtelASGIMiddleware)
    app.state._otel_instrumented = True


def autoinstrument(service_name: str | None = None) -> bool:
    """초기화 + 나가는 요청 전파 + 이후 생성되는 FastAPI 앱 자동 계측."""
    if not init_otel(service_name):
        return False
    _patch_clients()
    try:
        import fastapi
    except ImportError:
        return True
    if getattr(fastapi.FastAPI, "_otel_wrapped", False):
        return True
    orig_init = fastapi.FastAPI.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        instrument_app(self)

    fastapi.FastAPI.__init__ = __init__
    fastapi.FastAPI._otel_wrapped = True
    return True


@contextmanager
def stage(name: str, **attrs):
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)
        yield span


def current_trace_id() -> str:
    if tracer is None:
        return ""
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.trace_id else ""
