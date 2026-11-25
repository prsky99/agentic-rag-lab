"""게이트 호출 span (10.1절)."""
from __future__ import annotations

from . import semconv as S

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind

    tracer = trace.get_tracer("rag.pipeline")
except Exception:                                   # noqa: BLE001
    trace = None
    tracer = None


def traced_gate(client, gate: str, state, questions):
    """결정 모델 호출 하나를 gate.<이름> span으로 감싸고 판정·확률을 속성으로 남깁니다."""
    if tracer is None:
        return client.ask(state, questions)
    with tracer.start_as_current_span(f"gate.{gate}", kind=SpanKind.CLIENT) as sp:
        sp.set_attribute(S.GEN_AI_OPERATION, "decision")
        sp.set_attribute(S.RAG_GATE, gate)
        try:
            ans = client.ask(state, questions)
            q = next(iter(ans.values()))
            sp.set_attribute(S.RAG_GATE_VERDICT,
                             str(q.get("choice") or q.get("verdict") or "noul"))
            p = q.get("confidence") or q.get("noul")
            if p is not None:
                sp.set_attribute(S.RAG_GATE_P, float(p))
            return ans
        except Exception as e:
            sp.record_exception(e)
            sp.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
