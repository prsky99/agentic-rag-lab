# services/orchestrator/orchestrate.py
"""전체 오케스트레이터 — 6장 상태 기계의 참조 구현 (6.6·6.9절).

평가(RRSI)와 프로덕션(`services/orchestrator/main.py`)이 실제로 실행하는 정본은
`domains/rag/harness/pipeline.py`의 `RAGPipeline.answer_sync`입니다. 이 파일은 그 루프를
6장의 부품(rewrite·loop·escalate·deadline·injection·policy_store)으로 단계별로 풀어 쓴
것으로, 외부 호출을 전부 ``deps`` 객체로 받습니다. ``deps``가 제공해야 하는 것:

    route(question, tenant_tier, floor) -> Route
    retrieve(tenant_id, question, path, top_k, deadline_ms) -> list[passage]
    rerank(question, passages, top_n, deadline_ms) -> list[passage]
    answerability(question, passages, floor) -> {"verdict", "p_yes", "coverage"}
    answerability_with_fallback(question, passages, scores, floor) -> 같은 형식
    generate(question, ctx, deadline_ms) -> (answer, citations)
    faithfulness(answer, citations, floor) -> {"ok", "p_grounded", "severity"}
    escalate(question, evidence, g3) -> {"dest", "p"}
    enqueue_human(req, answer, evidence, g3) -> None
    decision (결정 모델 클라이언트, .ask), llm_cheap (.complete), dictionary (dict)

passage는 ``chunk_id``·``text``·``score`` 속성을 가진 객체입니다. 결정 모델 장애는
``system1.client.DecisionUnavailable``로 알립니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from services.guard.checks import CANARY, SECRET_RE
from system1.client import DecisionUnavailable

from .deadline import Deadline
from .injection import screen_passages
from .loop import should_retry
from .policy_store import PolicySnapshot
from .rewrite import plan_rewrite, rewrite


@dataclass(frozen=True)
class OrchestrateRequest:
    question: str
    tenant_id: str
    tenant_tier: str = "free"
    deadline_ms: int = 6000
    request_id: str = ""

    def with_question(self, question: str) -> "OrchestrateRequest":
        return replace(self, question=question)


@dataclass
class Route:
    path: str
    reason: str
    p: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class Evidence:
    ids: set[str]
    p_yes: float
    verdict: str
    coverage: float
    flags: set[str]        # {"partial_evidence", "stale_docs", "no_citation"}


@dataclass
class Result:
    status: str                  # answered | abstained | escalated
    answer: str | None = None
    citations: list = field(default_factory=list)
    reason: str = ""
    policy_version: str = ""
    stages: list = field(default_factory=list)


FLAG_INSTRUCTIONS = {
    "partial_evidence": "제공된 근거가 질문의 일부만 다룬다. 다루지 못한 부분은 모른다고 밝힌다.",
    "no_citation": "모든 문장 끝에 근거 번호 [n]을 붙인다.",
}


def build_context(passages: list, flags: set[str]) -> str:
    """판정 플래그를 지시로 바꿔 근거 앞에 붙입니다(6.6절: 판정이 프롬프트로 흘러가는 경로)."""
    notes = [FLAG_INSTRUCTIONS[f] for f in sorted(flags) if f in FLAG_INSTRUCTIONS]
    body = "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
    return "\n".join(notes + ["<context>", body, "</context>"])


def timed_out(dl: Deadline, policy: PolicySnapshot, stages: list, at: str) -> Result | None:
    """만료됐으면 기권 결과를, 아니면 None을 돌려줍니다. 각 호출 뒤와 반환 직전에 부릅니다.

    ``deps``의 호출은 ``deadline_ms``를 받아 자기 시간을 제한하지만, 그 값을 지키는지는
    구현에 달려 있습니다. 그래서 호출이 끝난 뒤에도 여기서 다시 검사합니다.
    """
    if not dl.expired():
        return None
    stages.append(("budget", f"deadline@{at}", 0))
    return Result("abstained", reason="deadline_exceeded",
                  policy_version=policy.version, stages=stages)


def orchestrate(req, deps, policy: PolicySnapshot) -> Result:
    dl = Deadline.start(req.deadline_ms)      # 0은 기본값으로 바꾸지 않습니다(이미 만료)
    ev = Evidence(ids=set(), p_yes=0.0, verdict="", coverage=0.0, flags=set())
    stages = []
    if (out := timed_out(dl, policy, stages, "start")):
        return out
    # 1) 라우팅
    try:
        route = deps.route(req.question, req.tenant_tier, policy.route_floor)
    except DecisionUnavailable:
        route = Route("local_docs", "fallback", 0.0, {})
    stages.append(("g1", route.path, dl.left_ms()))
    if route.path == "no_retrieval":
        return generate_and_verify(req, None, ev, dl, deps, policy, stages)
    # 2) 검색 + 평가 루프 (재작성 rewrite_max회 = 기본 1회 → 검색 시도 최대 2회)
    max_attempts = policy.rewrite_max + 1
    attempts, prev = 0, None
    while True:
        attempts += 1
        # 검색·리랭킹은 생성·G3·응답 조립 몫(3,500+400+450=4,350ms)을 남깁니다
        passages = deps.retrieve(req.tenant_id, req.question, route.path,
                                 top_k=50, deadline_ms=dl.slice_ms(300, 4350))
        passages = deps.rerank(req.question, passages, top_n=8,
                               deadline_ms=dl.slice_ms(200, 4350))
        if (out := timed_out(dl, policy, stages, "retrieve")):
            return out
        passages, dropped = screen_passages(deps.decision, passages)
        if dropped:
            ev.flags.add("injection_dropped")
        try:
            g2 = deps.answerability(req.question, passages, policy.answer_floor)
        except DecisionUnavailable:
            g2 = deps.answerability_with_fallback(
                req.question, passages, [p.score for p in passages], policy.answer_floor)
            ev.flags.add("g2_fallback")
        ev.ids = {p.chunk_id for p in passages}
        ev.p_yes, ev.verdict, ev.coverage = g2["p_yes"], g2["verdict"], g2.get("coverage", 0.0)
        stages.append(("g2", ev.verdict, dl.left_ms()))
        if g2["verdict"] == "sufficient":
            break
        if g2["verdict"] == "abstain":
            return Result("abstained", reason="insufficient_evidence",
                          policy_version=policy.version, stages=stages)
        cur = {"p_yes": ev.p_yes, "ids": set(ev.ids)}
        if attempts >= max_attempts:
            retry, why = False, "max_attempts"
        elif prev is None:
            retry, why = True, "first_rewrite"      # 비교 대상이 없는 첫 재작성
        else:
            retry, why = should_retry(prev, cur, attempts, max_attempts, dl.left_ms())
        if not retry:
            ev.flags.add(f"stop:{why}")
            return Result("abstained", reason=f"stop_{why}",
                          policy_version=policy.version, stages=stages)
        plan = plan_rewrite(deps.decision, req.question, {"p_yes": ev.p_yes})
        plan = rewrite(deps.llm_cheap, req.question, plan, deps.dictionary)
        # 요청은 불변 객체로 다룹니다. with_question은 질문만 바꾼 사본을 만듭니다.
        req = req.with_question(plan.queries[0])
        prev = cur
        stages.append(("rewrite", plan.strategy, dl.left_ms()))
    return generate_and_verify(req, passages, ev, dl, deps, policy, stages)


def guarded(result: Result, dl: Deadline | None = None) -> Result:
    """최종 응답 출구의 출력 가드(11.4절). answered만 검사하고, 걸리면 기권으로 바꿉니다.

    답변 본문과 인용 구간 모두에서 프롬프트 카나리·비밀 패턴을 찾습니다. ``dl``을 주면
    반환 직전에 만료도 검사해, 시간이 지난 답을 내보내지 않습니다.
    """
    if result.status != "answered":
        return result
    if dl is not None and dl.expired():
        result.stages.append(("budget", "deadline@finish", 0))
        return Result("abstained", reason="deadline_exceeded",
                      policy_version=result.policy_version, stages=result.stages)
    spans = [str(c.get("span", "") if isinstance(c, dict) else getattr(c, "span", ""))
             for c in (result.citations or [])]
    texts = [result.answer or ""] + spans
    why = ("prompt_leak" if any(CANARY in t for t in texts)
           else "secret_pattern" if any(SECRET_RE.search(t) for t in texts) else None)
    result.stages.append(("guard", f"block:{why}" if why else "ok", None))
    if why:
        return Result("abstained", reason=f"output_blocked_{why}",
                      policy_version=result.policy_version, stages=result.stages)
    return result


def generate_and_verify(req, passages, ev, dl, deps, policy, stages) -> Result:
    ctx = build_context(passages, ev.flags) if passages else None
    # 생성 뒤의 G3(400ms)와 응답 조립(450ms) 몫 850ms를 남깁니다(6.5절 예산 표)
    answer, cites = deps.generate(req.question, ctx,
                                  deadline_ms=dl.slice_ms(3500, 850))
    stages.append(("generate", "ok", dl.left_ms()))
    if passages is None:
        # no_retrieval 경로: 근거가 없는 것이 정상입니다. G3를 적용하지 않습니다.
        # 대신 생성 프롬프트가 사실 주장을 하지 않도록 제한하고, 응답에 표시를 남깁니다.
        return guarded(Result("answered", answer=answer, citations=[], reason="no_retrieval",
                              policy_version=policy.version, stages=stages), dl)
    if (out := timed_out(dl, policy, stages, "generate")):
        return out
    try:
        g3 = deps.faithfulness(answer, cites, policy.faith_floor)
    except DecisionUnavailable:
        return Result("abstained", reason="g3_unavailable",
                      policy_version=policy.version, stages=stages)
    stages.append(("g3", "ok" if g3["ok"] else "unsupported", dl.left_ms()))
    if g3["ok"]:
        return guarded(Result("answered", answer=answer, citations=cites,
                              policy_version=policy.version, stages=stages), dl)
    if g3["severity"] < 2.0 and dl.left_ms() > 1200:      # 가벼운 위반만 1회 재생성
        answer, cites = deps.generate(req.question, ctx, deadline_ms=dl.slice_ms(900, 850))
        try:
            g3 = deps.faithfulness(answer, cites, policy.faith_floor)
        except DecisionUnavailable:
            g3 = {"ok": False, "p_grounded": 0.0, "severity": 3.0}
        stages.append(("g3", "ok" if g3["ok"] else "unsupported_after_regen", dl.left_ms()))
        if g3["ok"]:
            return guarded(Result("answered", answer=answer, citations=cites,
                                  policy_version=policy.version, stages=stages), dl)
    try:
        esc = deps.escalate(req.question, {"p_yes": ev.p_yes,
                                           "tenant_tier": req.tenant_tier}, g3)
    except DecisionUnavailable:
        esc = {"dest": "abstain", "p": 0.0}
    stages.append(("g4", esc["dest"], dl.left_ms()))
    if esc["dest"] == "human":
        deps.enqueue_human(req, answer, ev, g3)
        return Result("escalated", reason="human_required",
                      policy_version=policy.version, stages=stages)
    # G3를 통과하지 못한 답은 answer/bigger_model 판정이어도 내보내지 않습니다
    return Result("abstained", reason="ungrounded", policy_version=policy.version, stages=stages)
