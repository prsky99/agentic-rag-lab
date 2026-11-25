# services/orchestrator/escalate.py
"""에스컬레이션 목적지 선택 (6.4절, 참조 구현).

정본은 `domains/rag/harness/gates.py`의 `escalate`(G4)입니다.
"""
from __future__ import annotations

ESCALATION_OPTIONS = {
    "answer": "현재 답변으로 충분하다. 그대로 전달한다.",
    "bigger_model": "근거는 충분하나 답변 품질이 낮다. 상위 모델로 재생성한다.",
    "human": "환불·법적 판단·계정 조치처럼 사람의 권한이 필요하다.",
    "abstain": "근거가 없거나 정책상 답할 수 없다. 기권한다.",
}


def choose_escalation(client, question: str, evidence: dict, answer: dict) -> dict:
    ans = client.ask(
        state={"question": question, "p_yes": evidence["p_yes"],
               "p_grounded": answer.get("p_grounded"),
               "severity": answer.get("severity"),
               "tenant_tier": evidence.get("tenant_tier", "free")},
        questions={"next": {"type": "choice",
                            "instructions": "이 요청을 다음 중 어디로 보내야 하는가?",
                            "criteria": ESCALATION_OPTIONS}},
    )
    n = ans["next"]
    return {"dest": n["choice"], "p": n["probabilities"][n["choice"]]}
