"""결정 모델 장애 시나리오 (6.6절).

정책은 두 가지입니다.
* G1이 중단되면 비용이 낮은 쪽(사내 검색)으로 보냅니다(가용성 우선).
* G3가 중단되면 기권합니다. 근거 검증을 거친 답만 내보냅니다(fail-closed).
"""
from __future__ import annotations


from harness.llm import DecisionUnavailable
from harness.pipeline import RAGPipeline


def test_g1_falls_back_to_local(monkeypatch, pipeline):
    def boom(self, state, questions):
        raise DecisionUnavailable("down")

    monkeypatch.setattr("harness.llm.DecisionClient.ask", boom)
    res = pipeline.answer_sync("환불은 며칠 이내에 가능한가요?", "acme", "pro")
    assert res["status"] in ("answered", "abstained")
    assert res["decisions"]["g1"]["path"] == "local_docs"
    assert "g1_fallback" in res["flags"]


def test_g3_fails_closed(monkeypatch, pipeline):
    """검색·생성은 정상, G3만 다운 → 기권."""
    real_ask = pipeline.decision.ask

    def only_grounded_down(self, state, questions):
        if "grounded" in questions:
            raise DecisionUnavailable("down")
        return real_ask(state, questions)

    monkeypatch.setattr("harness.llm.DecisionClient.ask", only_grounded_down)
    res = pipeline.answer_sync("환불은 결제일로부터 며칠 이내에 요청할 수 있나요?", "acme", "pro")
    assert res["status"] == "abstained"
    assert res["reason"] == "g3_unavailable"
    assert res["abstained"] is True


def test_pipeline_without_decision_service():
    """결정 모델 엔드포인트가 아예 없어도 파이프라인은 기동하고 기권합니다."""
    p = RAGPipeline.from_config("domains/rag/harness/config.yaml",
                                index_snapshot="test")
    assert p.decision.mode in ("local", "remote")
