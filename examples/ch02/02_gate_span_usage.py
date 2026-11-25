# examples/ch02/02_gate_span_usage.py
# 사용 예 — G2 충분성 게이트 호출
with stage("gate.answerability", **{"rag.tenant": tenant_id, "rag.gate": "g2"}) as span:
    verdict = sufficiency(client, question, passages, floor=G2_FLOOR)
    if span:
        span.set_attribute("rag.gate.p_yes", verdict["p_yes"])
        span.set_attribute("rag.gate.verdict", verdict["verdict"])
        span.set_attribute("rag.gate.threshold", G2_FLOOR)
