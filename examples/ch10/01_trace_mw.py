# examples/ch10/01_trace_mw.py
# gateway 미들웨어 (FastAPI, 개념 코드)
@app.middleware("http")
async def trace_mw(request, call_next):
    ctx = extract(request.headers)                 # W3C traceparent
    with tracer.start_as_current_span("rag.request", context=ctx) as sp:
        request.state.trace_id = format_trace_id(sp.get_span_context().trace_id)
        resp = await call_next(request)
        resp.headers["x-trace-id"] = request.state.trace_id
        return resp
