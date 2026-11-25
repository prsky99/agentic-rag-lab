# services/orchestrator/rewrite.py
"""질의 재작성: 전략 선택(결정 모델) + 재작성(생성 모델) (6.2절, 참조 구현).

정본 하네스(`domains/rag/harness/pipeline.py`의 `_rewrite`)는 전략 선택 없이
생성 모델 1회 재작성 또는 동의어 사전 치환만 합니다. 이 모듈은 6.2절의 네 전략을
풀어 쓴 확장형입니다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

STRATEGIES = {
    "normalize": "약어·표기·띄어쓰기 차이를 문서 표기로 맞춘다.",
    "expand": "사용자 표현을 문서의 동의어·전문 용어로 넓힌다.",
    "decompose": "여러 사실을 묻는 질문을 2~3개의 하위 질문으로 나눈다.",
    "constrain": "기간·제품·지역 조건을 메타데이터 필터로 바꾼다.",
}


@dataclass
class RewritePlan:
    strategy: str
    p: float
    queries: list[str]
    filters: dict


def plan_rewrite(client, question: str, failure: dict) -> RewritePlan:
    ans = client.ask(
        state={"question": question, "failure": failure},
        questions={
            "strategy": {
                "type": "choice",
                "instructions": "이 검색 실패를 고치는 데 가장 알맞은 재작성 전략은?",
                "criteria": STRATEGIES,
            },
            "multi_hop": {
                "type": "noul",
                "instructions": "이 질문은 서로 다른 두 사실을 결합해야 답할 수 있다.",
            },
        },
    )
    s = ans["strategy"]
    return RewritePlan(s["choice"], s["probabilities"][s["choice"]], [], {})


REWRITE_SYSTEM = """너는 검색 질의 재작성기다. 규칙:
- 원 질문의 의도를 바꾸지 않는다.
- 새로운 고유명사·숫자를 만들지 않는다. 사전에 있는 표기만 쓴다.
- 출력은 JSON만: {"queries": ["..."], "filters": {"key": "value"}}
- 질의는 최대 3개, 각 120자 이내."""


def rewrite(llm, question: str, plan: RewritePlan, dictionary: dict[str, str]) -> RewritePlan:
    if plan.strategy == "normalize":
        q = question
        for abbr, full in dictionary.items():
            q = re.sub(rf"\b{re.escape(abbr)}\b", full, q)
        return RewritePlan(plan.strategy, plan.p, [q], plan.filters)
    user = json.dumps({"question": question, "strategy": plan.strategy,
                       "allowed_terms": list(dictionary.values())[:200]},
                      ensure_ascii=False)
    raw = llm.complete(system=REWRITE_SYSTEM, user=user, temperature=0.0, max_tokens=300)
    try:
        obj = json.loads(raw)
        queries = [q.strip() for q in obj.get("queries", []) if q.strip()][:3]
        filters = obj.get("filters", {}) or {}
    except json.JSONDecodeError:
        return RewritePlan(plan.strategy, plan.p, [question], plan.filters)  # 원문 유지
    return RewritePlan(plan.strategy, plan.p, queries or [question], filters)
