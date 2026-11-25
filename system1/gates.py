"""게이트 4종 (1.4절, 4.4절).

G1 라우터 → G2 충분성 → G3 충실성 → G4 에스컬레이션.
각 함수는 결정 모델의 답을 임계값 정책으로 바꿉니다. 임계값은 하네스에서
진화하는 값이므로 여기서는 인자로 받습니다.

``sufficiency``는 1장의 스텁(문단 전체, 1,500자)이고, ``answerability``는
4장의 운영 버전(지연 예산에 맞춘 6문단 × 900자)입니다. 문단·인용은
``system1.types``의 ``Passage``/``Citation``을 씁니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# G3 심각도 상한. 정본 하네스(domains/rag/harness/gates.py)와 같은 정책입니다:
# p_grounded >= floor 그리고 severity < 2.0 이어야 통과합니다.
SEVERITY_CEILING = 2.0

ROUTE_OPTIONS = {
    "no_retrieval": "인사, 잡담, 또는 모델이 이미 아는 일반 상식. 사내 근거가 필요 없다.",
    "local_docs":   "사내 문서·정책·계약·매뉴얼에 근거가 있어야 답할 수 있다.",
    "web":          "최신 외부 정보(가격·일정·뉴스·공시)가 필요하다.",
}


@dataclass
class Route:
    path: str
    reason: str
    raw: dict = field(default_factory=dict)
    p: float = 0.0               # 선택된 경로의 확률(4장에서 로그·임계값 조정에 씁니다)


def route(client, question: str, tenant_tier: str = "free",
          floor: float = 0.60) -> Route:
    ans = client.ask(
        state={"question": question, "tenant_tier": tenant_tier},
        questions={
            "path": {"type": "choice",
                     "instructions": "이 질문에 답하려면 무엇을 검색해야 하는가?",
                     "criteria": ROUTE_OPTIONS},
            "needs_fresh": {"type": "noul",
                            "instructions": "정답이 최근 공개된 외부 정보에 달려 있다.",
                            "criteria": {
                                "true": "가격·일정·뉴스처럼 바뀌는 외부 사실이 답을 좌우한다.",
                                "false": "사내 문서나 일반 상식으로 답할 수 있다.",
                            }},
        },
    )
    p = ans["path"]
    top_p = p["probabilities"][p["choice"]]
    if top_p < floor:
        # 확률이 낮음 = 질문이 애매함. 비용이 낮은 기본 경로(사내 검색)로 보냅니다.
        return Route("local_docs", "low_confidence", ans, top_p)
    if ans["needs_fresh"]["noul"] > 0.7 and p["choice"] != "no_retrieval":
        return Route("web", "fresh_data", ans, top_p)
    return Route(p["choice"], "model_choice", ans, top_p)


def sufficiency(client, question: str, passages: list, floor: float = 0.65) -> dict:
    state = {
        "question": question,
        "passages": [{"id": p.doc_id, "text": p.text[:1500]} for p in passages],
    }
    ans = client.ask(state, {
        "enough": {
            "type": "noul",
            "instructions": "주어진 근거만으로 질문에 답할 수 있다.",
            "criteria": {
                "true":  "근거 문단에 답이 명시되어 있거나, 문단 간 결합으로 확정할 수 있다.",
                "false": "답이 없거나 일부만 있어 추측이 필요하다.",
            },
        },
        "coverage": {
            "type": "score",
            "instructions": "질문의 요구사항을 근거가 얼마나 덮는가?",
            "criteria": ["거의 없음", "일부", "절반", "대부분", "완전"],
        },
    })
    p_yes = ans["enough"]["noul"]
    if p_yes >= floor:
        verdict = "sufficient"
    elif 0.3 <= p_yes < floor:
        verdict = "rewrite"          # 질의를 바꿔 1회만 재검색
    else:
        verdict = "abstain"          # 근거 없음 → 기권
    return {"verdict": verdict, "p_yes": p_yes, "score": ans["coverage"]["score"]}


def answerability(client, question: str, passages: list, floor: float = 0.65) -> dict:
    """G2 운영 버전. state 크기가 지연을 결정합니다. 문단당 900자, 최대 6개로 제한."""
    state = {
        "question": question,
        "passages": [{"id": p.chunk_id, "text": p.text[:900]} for p in passages[:6]],
    }
    ans = client.ask(state, {
        "enough": {
            "type": "noul",
            "instructions": "주어진 근거만으로 질문에 답할 수 있다.",
            "criteria": {
                "true":  "근거 문단에 답이 명시되어 있거나, 문단 간 결합으로 확정할 수 있다.",
                "false": "답이 없거나 일부만 있어 추측이 필요하다.",
            },
        },
        "coverage": {
            "type": "score",
            "instructions": "질문의 요구사항을 근거가 얼마나 덮는가?",
            "criteria": ["거의 없음", "일부", "절반", "대부분", "완전"],
        },
    })
    p_yes = ans["enough"]["noul"]
    if p_yes >= floor:
        verdict = "sufficient"
    elif 0.3 <= p_yes < floor:
        verdict = "rewrite"
    else:
        verdict = "abstain"
    return {"verdict": verdict, "p_yes": p_yes, "coverage": ans["coverage"]["score"]}


def faithfulness(client, answer: str, citations: list, floor: float = 0.70) -> dict:
    state = {
        "answer": answer,
        "citations": [{"id": c.chunk_id, "span": c.span, "text": c.text[:700]}
                      for c in citations[:6]],
    }
    ans = client.ask(state, {
        "grounded": {
            "type": "noul",
            "instructions": "답변의 모든 사실 주장이 인용된 근거로 뒷받침된다.",
            "criteria": {
                "true":  "모든 주장이 인용 문단에 있거나, 문단에서 직접 도출된다.",
                "false": "인용에 없는 주장이 하나 이상 있거나, 인용이 주장과 다른 내용을 가리킨다.",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "근거 없는 주장이 답변 전체에서 차지하는 비중은?",
            "criteria": ["없음", "한 조각", "일부", "상당 부분", "대부분"],
        },
    })
    p = ans["grounded"]["noul"]
    sev = ans["severity"]["score"]
    # 확률이 임계값을 넘어도 severity가 높으면 보수적으로 판단합니다
    ok = p >= floor and sev < SEVERITY_CEILING
    return {"ok": ok, "p_grounded": p, "severity": sev}


ESCALATION_OPTIONS = {
    "answer": "현재 답변으로 충분하다. 그대로 전달한다.",
    "bigger_model": "근거는 충분하나 답변 품질이 낮다. 상위 모델로 재생성한다.",
    "human": "환불·법적 판단·계정 조치처럼 사람의 권한이 필요하다.",
    "abstain": "근거가 없거나 정책상 답할 수 없다. 기권한다.",
}


def escalate(client, question: str, evidence: dict, answer: dict) -> dict:
    ans = client.ask(
        state={"question": question, "p_yes": evidence.get("p_yes"),
               "p_grounded": answer.get("p_grounded"),
               "severity": answer.get("severity"),
               "tenant_tier": evidence.get("tenant_tier", "free")},
        questions={"next": {"type": "choice",
                            "instructions": "이 요청을 다음 중 어디로 보내야 하는가?",
                            "criteria": ESCALATION_OPTIONS}},
    )
    n = ans["next"]
    return {"dest": n["choice"], "p": n["probabilities"][n["choice"]]}
