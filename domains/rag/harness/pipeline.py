"""제어 흐름: G1 → 검색 → 리랭킹 → G2 → 생성 → G3 → G4 (6장, 8.1절).

이 파일이 하네스의 중심이고, RRSI가 편집하는 대상입니다. 평가와 프로덕션이
같은 코드를 씁니다. ``run.py``가 평가에서 subprocess로 부르고, orchestrator가
HTTP 핸들러 안에서 같은 클래스를 부릅니다.

요청 한 건의 상태(설정 사본·데드라인·호출 장부·판정 기록)는 ``_Run``에 담고
요청끼리 공유하지 않습니다. 그래서 등급별 정책이 다른 요청이 동시에 들어와도
설정이 섞이지 않습니다.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from . import gates
from .budget import BudgetExceeded, CallLedger, MeteredDecision
from .context import Context, Citation, body_of, build_context, sentences
from .guard import output_block_reason
from .llm import (DecisionClient, DecisionUnavailable, LLMClient, LLMUnavailable,
                  call_scope, estimate_tokens, looks_like_injection, weighted_overlap)
from .retrieve import Passage, RetrievalUnavailable, build_retriever, rerank

HERE = Path(__file__).resolve().parent
PROMPT_DIR = HERE / "prompts"
ABSTAIN_TEXT = "문서에서 찾지 못했습니다."
MARKER = re.compile(r"\[(\d+)\]")
POLICY_SECTIONS = ("gates", "retrieval", "rerank", "context", "budget")


@dataclass(frozen=True)
class AnswerRequest:
    question: str
    tenant: str
    tenant_tier: str = "free"
    deadline_ms: int = 6000

    def with_question(self, question: str) -> "AnswerRequest":
        return replace(self, question=question)


@dataclass
class Deadline:
    started: float
    total_ms: int

    @classmethod
    def start(cls, total_ms: int) -> "Deadline":
        return cls(time.monotonic(), total_ms)

    def left_ms(self) -> int:
        return max(0, self.total_ms - int((time.monotonic() - self.started) * 1000))

    def slice_ms(self, budget_ms: int, reserve_ms: int = 0) -> int:
        return max(0, min(budget_ms, self.left_ms() - reserve_ms))

    def expired(self) -> bool:
        return self.left_ms() <= 0


@dataclass
class Evidence:
    ids: set[str] = field(default_factory=set)
    p_yes: float = 0.0
    verdict: str = ""
    coverage: float = 0.0
    flags: set[str] = field(default_factory=set)
    context_ids: list[str] = field(default_factory=list)   # 리랭킹·주입 스크리닝 후 순서


@dataclass
class _Run:
    """요청 한 건의 실행 상태. 요청마다 새로 만들고 다른 요청과 공유하지 않습니다."""
    tier: str
    gates_cfg: dict
    ret_cfg: dict
    rerank_cfg: dict
    ctx_cfg: dict
    budget: dict
    deadline: Deadline
    ledger: CallLedger
    ev: Evidence
    t0: float
    decisions: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)

    def gate(self, stage: str, client) -> MeteredDecision:
        return MeteredDecision(client, self.ledger, stage)

    def policy(self) -> dict:
        """이 요청에 실제로 적용한 정책 값(응답의 ``policy``)."""
        return {"tier": self.tier,
                "top_k": int(self.ret_cfg.get("top_k", 40)),
                "max_rewrites": int(self.gates_cfg.get("max_rewrites", 1)),
                "g2_floor": float(self.gates_cfg.get("g2_floor", 0.65)),
                "max_gate_calls": self.ledger.max_gate_calls,
                "max_generation_calls": self.ledger.max_generation_calls,
                "deadline_ms": self.deadline.total_ms}


def _read_prompts() -> dict[str, str]:
    out = {}
    for name in ("system", "rewrite", "cite"):
        path = PROMPT_DIR / f"{name}.md"
        out[name] = path.read_text(encoding="utf-8") if path.exists() else ""
    return out


class RAGPipeline:
    def __init__(self, cfg: dict, retriever, decision: DecisionClient,
                 llm: LLMClient, prompts: dict[str, str], seed: int = 0,
                 index_snapshot: str | None = None, config_dir: Path | None = None):
        self.cfg = cfg or {}
        self.retriever = retriever
        self.decision = decision
        self.llm = llm
        self.prompts = prompts
        self.seed = seed
        self.index_snapshot = index_snapshot
        self.config_dir = config_dir or HERE
        self.gates_cfg = dict(self.cfg.get("gates") or {})
        self.ret_cfg = dict(self.cfg.get("retrieval") or {})
        self.rerank_cfg = dict(self.cfg.get("rerank") or {})
        self.ctx_cfg = dict(self.cfg.get("context") or {})
        self.budget = dict(self.cfg.get("budget") or {})
        self.policy_version = str((self.cfg.get("policy") or {}).get("version", "rag-harness"))

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, path: str | Path | None = None, seed: int = 0,
                    index_snapshot: str | None = None) -> "RAGPipeline":
        cfg_path = Path(path) if path else HERE / "config.yaml"
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path) if cfg_path.exists() else (HERE / cfg_path)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        retriever = build_retriever(dict(cfg.get("retrieval") or {}), cfg_path.parent)
        gen = dict(cfg.get("generation") or {})
        decision = DecisionClient(base_url=gen.get("decision_base_url") or None,
                                  model=gen.get("decision_model") or None,
                                  mode=str(gen.get("decision_mode", "auto")))
        llm = LLMClient(base_url=gen.get("base_url") or None, model=gen.get("model") or None,
                        api_key=gen.get("api_key") or None,
                        mode=str(gen.get("mode", "auto")),
                        provider=gen.get("provider") or None)
        return cls(cfg, retriever, decision, llm, _read_prompts(), seed=seed,
                   index_snapshot=index_snapshot or gen.get("index_snapshot"),
                   config_dir=cfg_path.parent)

    def runtime(self) -> dict:
        """이 파이프라인이 실제로 쓰는 실행 조건(평가 지문의 재료, E03).

        설정 파일과 환경 변수를 ``from_config``가 해석한 **결과**를 읽습니다. 환경 변수 이름을
        나열하지 않으므로 우선순위(설정 값 → 환경 변수 → 기본값)가 그대로 반영됩니다.
        같은 job 이름이라도 이 값이 다르면 평가 어댑터는 이전 산출물을 다시 쓰지 않습니다.
        키·토큰 값은 넣지 않고, 있는지만 남깁니다.
        """
        dec, gen, ret = self.decision, self.llm, self.retriever
        passages = getattr(ret, "passages", None)
        retrieval = {"backend": "local" if passages is not None else "http",
                     "address": getattr(ret, "base_url", "")}
        if passages is not None:
            # 로컬 색인의 내용 해시: 청크 ID와 본문이 바뀌면 값이 바뀝니다.
            h = hashlib.sha256()
            for p in passages:
                h.update(f"{p.chunk_id}\0{p.tenant}\0{p.text}\0".encode("utf-8"))
            retrieval["index_sha256"] = h.hexdigest()[:16]
            retrieval["chunks"] = len(passages)
        snap = str(self.index_snapshot or "")
        snap_file = self.config_dir.parent / "data" / "index" / f"{snap}.json"
        cfg = json.loads(json.dumps(self.cfg, ensure_ascii=False, default=str))
        if isinstance(cfg.get("generation"), dict) and "api_key" in cfg["generation"]:
            cfg["generation"]["api_key"] = bool(cfg["generation"]["api_key"])
        return {
            "decision": {"mode": str(getattr(dec, "mode", "")),          # local|remote|offline
                         "model": str(getattr(dec, "model", "")),
                         "address": str(getattr(dec, "base_url", "")),
                         "provider": "jev" if getattr(dec, "mode", "") == "remote"
                         else "local_fallback"},
            "generation": {"mode": str(getattr(gen, "mode", "")),        # remote|extractive
                           "provider": str(getattr(gen, "provider", "")),
                           "model": str(getattr(gen, "model", "")),
                           "address": str(getattr(gen, "base_url", "")),
                           "has_api_key": bool(getattr(gen, "api_key", ""))},
            "retrieval": retrieval,
            "index_snapshot": {"name": snap,
                               "sha256": hashlib.sha256(snap_file.read_bytes()).hexdigest()[:16]
                               if snap and snap_file.is_file() else ""},
            "policy_version": self.policy_version,
            # config.yaml을 읽은 최종 값 전체입니다(키 값은 있는지만 남깁니다).
            "config": cfg,
        }

    # ------------------------------------------------------------------
    async def answer(self, question: str, tenant: str, tenant_tier: str = "free",
                     deadline_ms: int | None = None, policy: dict | None = None) -> dict:
        # 실제 I/O는 동기 라이브러리(urllib)를 씁니다. 이벤트 루프를 막지 않도록 작업 스레드에서
        # 실행합니다(FastAPI 핸들러·평가 하네스의 asyncio.gather가 동시에 여러 질의를 보냅니다).
        return await asyncio.to_thread(self.answer_sync, question, tenant, tenant_tier,
                                       deadline_ms, policy)

    def answer_sync(self, question: str, tenant: str, tenant_tier: str = "free",
                    deadline_ms: int | None = None, policy: dict | None = None) -> dict:
        """``policy``는 이 요청에만 덮어쓸 설정 절입니다(config.yaml과 같은 구조).

        orchestrator가 인증된 등급에서 계산한 정책(``services/gateway/tier.py``)을 넘깁니다.
        ``self.cfg``는 바꾸지 않으므로 동시 요청끼리 정책이 섞이지 않습니다.
        """
        ev = Evidence()
        run = self._new_run(tenant_tier, deadline_ms, ev, policy)
        try:
            res = self._answer_impl(question, tenant, tenant_tier, run)
        except BudgetExceeded as exc:
            ev.flags.add(f"budget:{exc.reason}@{exc.stage}")
            run.stages.append(["budget", f"{exc.reason}@{exc.stage}", run.deadline.left_ms()])
            reason = "deadline_exceeded" if exc.reason == "deadline" else "call_budget_exceeded"
            res = self._finish(run, "abstained", "", [], True, reason=reason)
        # retrieved는 리랭킹 전 검색 순서, context_ids는 리랭킹 후 순서입니다(9.3절 평가용).
        res["context_ids"] = list(ev.context_ids)
        return res

    def _request_cfg(self, policy: dict | None) -> dict:
        base = {"gates": self.gates_cfg, "retrieval": self.ret_cfg, "rerank": self.rerank_cfg,
                "context": self.ctx_cfg, "budget": self.budget}
        out = {}
        for sec in POLICY_SECTIONS:
            merged = dict(base[sec])                 # 사본: 공유 설정을 바꾸지 않습니다
            merged.update(dict((policy or {}).get(sec) or {}))
            out[sec] = merged
        return out

    def _new_run(self, tenant_tier: str, deadline_ms: int | None, ev: Evidence,
                 policy: dict | None) -> _Run:
        rc = self._request_cfg(policy)
        budget = rc["budget"]
        # None만 기본값으로 바꿉니다. 0(또는 음수)은 "이미 만료"이므로 첫 검사에서 기권합니다.
        if deadline_ms is None:
            deadline_ms = budget.get("deadline_ms", 6000)
        deadline = Deadline.start(max(0, int(deadline_ms)))
        ledger = CallLedger(deadline, int(budget.get("max_gate_calls", 5)),
                            int(budget.get("max_generation_calls", 2)))
        return _Run(tier=tenant_tier, gates_cfg=rc["gates"], ret_cfg=rc["retrieval"],
                    rerank_cfg=rc["rerank"], ctx_cfg=rc["context"], budget=budget,
                    deadline=deadline, ledger=ledger, ev=ev, t0=time.monotonic())

    def _answer_impl(self, question: str, tenant: str, tenant_tier: str, run: _Run) -> dict:
        ev, decisions, stages, t0 = run.ev, run.decisions, run.stages, run.t0
        question = gates.sanitize(question)
        run.ledger.check_deadline("start")         # deadline_ms=0은 여기서 끝납니다

        # 질문 자체가 지시 탈취를 시도하면 검색·생성 없이 바로 기권합니다.
        if looks_like_injection(question):
            ev.flags.add("injection_in_question")
            return self._finish(run, "abstained", "", [], True, reason="injection_in_question")

        # 1) G1 라우팅
        try:
            r = gates.route(run.gate("g1", self.decision), question, tenant_tier,
                            float(run.gates_cfg.get("g1_floor", 0.60)))
        except DecisionUnavailable:
            ev.flags.add("g1_fallback")
            r = {"path": "local_docs", "reason": "decision_unavailable", "p": 0.0}
        decisions["g1"] = {**r, "ms": deadlines_ms(t0)}
        stages.append(["g1", r["path"], run.deadline.left_ms()])

        if r["path"] == "no_retrieval":
            return self._generate_and_verify(
                AnswerRequest(question, tenant, tenant_tier), None, run, reason="no_retrieval")

        search = self.retriever.search
        if r["path"] == "web":
            web_search = getattr(self.retriever, "search_web", None)
            if web_search is not None:
                search = web_search
                decisions["g1"]["served_path"] = "web"
            else:
                # 웹 어댑터가 없습니다. 조용히 로컬 검색으로 바꾸지 않고 결과에 표시합니다.
                ev.flags.add("web_unsupported")
                if str(run.ret_cfg.get("web_policy", "local_fallback")) == "abstain":
                    decisions["g1"]["served_path"] = "none"
                    return self._finish(run, "abstained", "", [], True, reason="web_unsupported")
                ev.flags.add("web_fallback_local")
                decisions["g1"]["served_path"] = "local_docs"
                stages.append(["route", "web_unsupported->local_docs", run.deadline.left_ms()])

        g2_floor = float(run.gates_cfg.get("g2_floor", 0.65))
        band = tuple(run.gates_cfg.get("g2_rewrite_band", [0.30, g2_floor]))
        max_rewrites = int(run.gates_cfg.get("max_rewrites", 1))

        # 2) 검색 → 리랭킹 → 주입 스크리닝 → G2 (재검색 상한 1회)
        attempts, prev = 0, None
        passages: list[Passage] = []
        while True:
            run.ledger.check_deadline("retrieve")
            t_ret = time.monotonic()
            try:
                with call_scope(run.ledger.timeout_s()):
                    passages = search(question, tenant,
                                      top_k=int(run.ret_cfg.get("top_k", 40)))
            except (RetrievalUnavailable, TimeoutError) as exc:
                # 검색 실패·시간 초과는 예외로 올리지 않고 통제된 기권으로 끝냅니다(RT04).
                run.ledger.check_deadline("retrieve")      # 만료가 원인이면 deadline_exceeded
                ev.flags.add("retrieval_unavailable")
                stages.append(["retrieve", "unavailable", run.deadline.left_ms()])
                res = self._finish(run, "abstained", "", [], True,
                                   reason="retrieval_unavailable")
                # 평가에서는 정상 기권이 아니라 실행 오류로 셉니다(E05).
                res["error"] = f"retrieval_unavailable: {exc}"[:300]
                return res
            run.ledger.check_deadline("retrieve")
            run.retrieved = [p.chunk_id for p in passages]
            top_n = int(run.rerank_cfg.get("top_n", 20)) if run.rerank_cfg.get("enabled", True) \
                else int(run.ctx_cfg.get("final_k", 8))
            # 리랭킹은 프로세스 안 연산이라 도중에 끊지 못합니다. 끝난 뒤 만료를 검사합니다.
            passages = rerank(question, passages, top_n, run.rerank_cfg)
            run.ledger.check_deadline("rerank")
            kept, dropped, screen_fallback = gates.screen_passages(
                run.gate("screen", self.decision), passages)
            if dropped:
                ev.flags.add("injection_dropped")
            if screen_fallback:
                ev.flags.add("injection_pattern_fallback")
            passages = kept
            ev.context_ids = [p.chunk_id for p in passages]
            stages.append(["retrieve", f"{len(passages)} passages",
                           run.deadline.left_ms(), int((time.monotonic() - t_ret) * 1000)])

            try:
                g2 = gates.answerability(run.gate("g2", self.decision), question, passages,
                                         g2_floor, band)
            except DecisionUnavailable:
                ev.flags.add("g2_fallback")
                g2 = gates.answerability_with_fallback(
                    question, passages, [p.score for p in passages], g2_floor)
            decisions["g2"] = {k: g2[k] for k in ("p_yes", "verdict")}
            decisions["g2"]["coverage"] = round(float(g2.get("coverage", 0.0)), 3)
            decisions["g2"]["ms"] = deadlines_ms(t0)

            ev.ids = {p.chunk_id for p in passages}
            ev.p_yes, ev.verdict = float(g2["p_yes"]), str(g2["verdict"])
            ev.coverage = float(g2.get("coverage", 0.0))
            stages.append(["g2", ev.verdict, run.deadline.left_ms()])

            if ev.verdict == "sufficient":
                break
            if ev.verdict == "abstain":
                return self._finish(run, "abstained", "", [], True,
                                    reason="insufficient_evidence")
            if prev is None:
                # 첫 재작성도 상한을 따릅니다(max_rewrites: 0이면 재작성 없이 기권).
                retry, why = (attempts < max_rewrites,
                              "first_rewrite" if attempts < max_rewrites else "max_attempts")
            else:
                retry, why = gates.should_retry(prev, {"p_yes": ev.p_yes, "ids": ev.ids},
                                                attempts, max_rewrites, run.deadline.left_ms())
            if not retry:
                ev.flags.add(f"stop:{why}")
                return self._finish(run, "abstained", "", [], True, reason=f"stop_{why}")
            prev = {"p_yes": ev.p_yes, "ids": set(ev.ids)}
            question = self._rewrite(question, run)
            attempts += 1
            stages.append(["rewrite", question[:40], run.deadline.left_ms()])

        # 3) 컨텍스트 조립 → 생성 → G3 → G4
        return self._generate_and_verify(
            AnswerRequest(question, tenant, tenant_tier), passages, run)

    # ------------------------------------------------------------------
    def _generate_and_verify(self, req: AnswerRequest, passages: list[Passage] | None,
                             run: _Run, reason: str = "") -> dict:
        ev, decisions, stages, t0 = run.ev, run.decisions, run.stages, run.t0
        g3_floor = float(run.gates_cfg.get("g3_floor", 0.70))
        ctx = build_context(passages or [], req.question,
                            {**run.ctx_cfg,
                             "final_k": run.ret_cfg.get("final_k", 8)}) \
            if passages else Context("", [], [])

        answer, citations, gen_mode = self._generate(req.question, ctx, run, "generate")
        stages.append(["generate", gen_mode, run.deadline.left_ms()])

        if not passages:
            # no_retrieval 경로: 근거가 없는 것이 정상입니다. G3는 건너뜁니다(출력 가드는 적용).
            return self._finish(run, "answered", answer, [], False,
                                reason=reason or "no_retrieval")

        try:
            g3 = gates.faithfulness(run.gate("g3", self.decision), answer, citations, g3_floor)
        except DecisionUnavailable:
            ev.flags.add("g3_unavailable")
            return self._finish(run, "abstained", "", [], True, reason="g3_unavailable")
        decisions["g3"] = {"p_grounded": g3["p_grounded"], "ok": g3["ok"],
                           "severity": g3["severity"], "ms": deadlines_ms(t0)}
        stages.append(["g3", "ok" if g3["ok"] else "unsupported", run.deadline.left_ms()])

        if g3["ok"]:
            return self._finish(run, "answered", answer, citations, False)

        # 가벼운 위반만 1회 재생성
        if g3["severity"] < gates.SEVERITY_CEILING and run.deadline.left_ms() > 1200:
            answer, citations, _ = self._generate(req.question, ctx, run, "regenerate",
                                                  temperature=0.0)
            try:
                g3 = gates.faithfulness(run.gate("g3", self.decision), answer, citations,
                                        g3_floor)
            except DecisionUnavailable:
                g3 = {"ok": False, "p_grounded": 0.0, "severity": 3.0}
            decisions["g3"] = {"p_grounded": g3["p_grounded"], "ok": g3["ok"],
                               "severity": g3["severity"], "ms": deadlines_ms(t0),
                               "regenerated": True}
            stages.append(["g3", "ok" if g3["ok"] else "unsupported_after_regen",
                           run.deadline.left_ms()])
            if g3["ok"]:
                return self._finish(run, "answered", answer, citations, False)

        try:
            esc = gates.escalate(run.gate("g4", self.decision), req.question,
                                 {"p_yes": ev.p_yes, "tenant_tier": req.tenant_tier}, g3)
        except DecisionUnavailable:
            esc = {"dest": "abstain", "p": 0.0}
        decisions["g4"] = {"dest": esc["dest"], "p": esc["p"], "ms": deadlines_ms(t0)}
        stages.append(["g4", esc["dest"], run.deadline.left_ms()])
        if esc["dest"] == "human":
            return self._finish(run, "escalated", "", [], True, reason="human_required")
        return self._finish(run, "abstained", "", [], True, reason="ungrounded")

    # ------------------------------------------------------------------
    def _generate(self, question: str, ctx: Context, run: _Run, stage: str,
                  temperature: float = 0.0) -> tuple[str, list[Citation], str]:
        run.ledger.begin(stage, generation=True)
        system = self.prompts.get("system", "") + "\n\n" + self.prompts.get("cite", "")
        user = json.dumps({"question": question, "context": ctx.text}, ensure_ascii=False)
        if self.llm.mode == "remote" and ctx.text:
            try:
                text, usage = self._llm_call(run, stage, system, user, temperature, 700)
                return text.strip(), self._citations_from(text, ctx), "llm"
            except LLMUnavailable:
                pass
        if not ctx.text:
            answer, cites, mode = self._chitchat(question), [], "template"
        else:
            answer, cites = self._extractive(question, ctx)
            mode = "extractive"
        # 모델 호출이 없는 대체 생성은 예산 배분용 추정치로만 기록합니다.
        run.ledger.record(stage, "extractive", estimate_tokens(ctx.text + question),
                          estimate_tokens(answer), "estimate")
        return answer, cites, mode

    def _llm_call(self, run: _Run, stage: str, system: str, user: str,
                  temperature: float, max_tokens: int) -> tuple[str, dict]:
        """생성 모델 호출 1회. 남은 시간을 타임아웃으로 넘기고 사용량을 장부에 남깁니다."""
        run.ledger.check_deadline(stage)
        try:
            with call_scope(run.ledger.timeout_s()):
                fn = getattr(self.llm, "complete_with_usage", None)
                if fn is not None:
                    text, usage = fn(system, user, temperature, max_tokens)
                else:
                    text = self.llm.complete(system=system, user=user,
                                             temperature=temperature, max_tokens=max_tokens)
                    usage = {"input_tokens": estimate_tokens(system + user),
                             "output_tokens": estimate_tokens(text), "source": "estimate"}
        except LLMUnavailable:
            # 실패한 호출은 미측정으로 남깁니다. 요청은 보냈을 수 있으므로 0으로 세지 않고,
            # 입력 글자 수 추정치를 따로 적습니다(대체 생성은 따로 기록됩니다).
            run.ledger.record_unknown(stage, "llm", estimate_tokens(system + user), "failed")
            run.ledger.check_deadline(stage)
            raise
        if usage.get("source") == "provider":
            run.ledger.record(stage, "llm", usage.get("input_tokens", 0),
                              usage.get("output_tokens", 0), "provider")
        else:
            run.ledger.record_unknown(
                stage, "llm", usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                str(usage.get("issue") or "missing"))
        # 응답을 받는 동안 시간이 다 됐으면 그 답을 쓰지 않습니다(RT04).
        run.ledger.check_deadline(stage)
        return text, usage

    def _citations_from(self, text: str, ctx: Context) -> list[Citation]:
        by_marker = {c.marker: c for c in ctx.citations}
        seen, out = set(), []
        for m in MARKER.findall(text):
            marker = int(m)
            if marker in by_marker and marker not in seen:
                seen.add(marker)
                out.append(by_marker[marker])
        return out

    def _extractive(self, question: str, ctx: Context) -> tuple[str, list[Citation]]:
        """생성 모델이 없을 때의 추출형 답변.

        검색된 문단에서 질문과 가장 겹치는 문장을 고르고 인용 표식을 붙입니다.
        없는 내용을 만들지 않는다는 점에서 이 모드의 답변은 항상 근거에 묶입니다.
        """
        scored: list[tuple[float, Citation, str]] = []
        by_chunk = {c.chunk_id: c for c in ctx.citations}
        # 문단 점수가 아니라 문장 점수로 고릅니다. 리랭커는 문단 단위로 정렬하지만
        # 답이 되는 문장은 그 문단 안의 한 줄인 경우가 많습니다.
        top_passages = sorted(ctx.used, key=lambda p: -p.score)
        idf = getattr(self.retriever, "idf", lambda t: 1.0)
        for p in top_passages:
            c = by_chunk.get(p.chunk_id)
            if not c:
                continue
            for s in sentences(body_of(p)):
                s = s.strip(" -•\t#")
                if len(s) < 6:
                    continue
                score = weighted_overlap(question, s, idf) \
                    + (0.08 if any(ch.isdigit() for ch in s) else 0.0)
                scored.append((score, c, s))
        scored.sort(key=lambda x: -x[0])
        chosen, seen = [], set()
        if not scored:
            return ABSTAIN_TEXT, []
        best_c = scored[0][1]
        floor = scored[0][0] * 0.5
        for score, c, s in scored:
            if score <= 0.0 or score < floor or c.chunk_id != best_c.chunk_id or s in seen:
                continue
            seen.add(s)
            chosen.append((c, s))
            if len(chosen) >= 2:
                break
        if not chosen and ctx.citations:
            c = ctx.citations[0]
            text = next((p.text for p in ctx.used if p.chunk_id == c.chunk_id), c.span)
            chosen = [(c, sentences(text)[0] if sentences(text) else c.span)]

        if not chosen:
            return ABSTAIN_TEXT, []
        # 문단 순서를 유지해 인용 순서를 보존합니다
        chosen.sort(key=lambda cs: cs[0].marker)
        answer = " ".join(f"{s} [{c.marker}]" for c, s in chosen)
        # 인용의 span은 실제로 답변에 쓴 문장입니다. best_span을 그대로 쓰면
        # 인용 구간과 답변 문장이 어긋나 충실성 판정이 근거 없이 흔들립니다.
        cites = [Citation(marker=c.marker, doc_id=c.doc_id, chunk_id=c.chunk_id, span=s)
                 for c, s in chosen]
        return answer, cites

    def _chitchat(self, question: str) -> str:
        if any(k in question for k in ("안녕", "반가")):
            return "안녕하세요. 문서를 근거로 답해 드립니다."
        if any(k in question for k in ("고마", "감사")):
            return "천만에요. 더 필요한 것이 있으면 말씀해 주세요."
        return "일반 상식으로 답할 수 있는 질문입니다."

    def _rewrite(self, question: str, run: _Run | None = None) -> str:
        """질의 재작성 1회. 모델이 없으면 동의어 사전만 적용합니다."""
        synonyms = dict((self.cfg.get("rewrite") or {}).get("synonyms") or {})
        if self.llm.mode == "remote":
            system = self.prompts.get("rewrite", "")
            user = json.dumps({"question": question,
                               "allowed_terms": list(synonyms.values())}, ensure_ascii=False)
            try:
                if run is not None:
                    raw, _ = self._llm_call(run, "rewrite", system, user, 0.0, 200)
                else:
                    raw = self.llm.complete(system=system, user=user, temperature=0.0,
                                            max_tokens=200)
                match = re.search(r"\{.*\}", raw, flags=re.S)
                if match:
                    queries = json.loads(match.group(0)).get("queries") or []
                    if queries and isinstance(queries[0], str):
                        return queries[0][:300]
            except (LLMUnavailable, json.JSONDecodeError, KeyError, AttributeError):
                pass
        out = question
        for abbr, full in synonyms.items():
            out = re.sub(rf"\b{re.escape(abbr)}\b", full, out)
        return out if out != question else question

    def _finish(self, run: _Run, status: str, answer: str, citations: list[Citation],
                abstained: bool, reason: str = "") -> dict:
        """모든 최종 응답이 지나는 출구. ``answered``는 출력 가드를 통과해야 나갑니다."""
        if status == "answered":
            why = output_block_reason(answer, [c.span for c in citations])
            run.stages.append(["guard", f"block:{why}" if why else "ok",
                               run.deadline.left_ms()])
            if why:
                run.ev.flags.add("output_blocked")
                status, answer, citations, abstained = "abstained", "", [], True
                reason = f"output_blocked_{why}"
        if status == "answered" and run.deadline.expired():
            # 최종 반환 직전의 만료 검사: 시간이 지난 답은 내보내지 않습니다(RT04).
            run.ev.flags.add("budget:deadline@finish")
            run.stages.append(["budget", "deadline@finish", 0])
            status, answer, citations, abstained = "abstained", "", [], True
            reason = "deadline_exceeded"
        return self._result(status, answer, citations, abstained, reason=reason,
                            decisions=run.decisions, stages=run.stages,
                            usage=run.ledger.usage(), retrieved=run.retrieved,
                            latency=deadlines_ms(run.t0),
                            flags=run.ev.flags | run.ledger.flags,
                            policy=run.policy())

    def _result(self, status: str, answer: str, citations: list[Citation],
                abstained: bool, reason: str = "", decisions: dict | None = None,
                stages: list | None = None, usage: dict | None = None,
                retrieved: list[str] | None = None, latency: int = 0,
                flags: set[str] | None = None, policy: dict | None = None) -> dict:
        usage = dict(usage or {})
        # 미측정 호출(usage_known=false)이 있으면 이 합계는 최솟값입니다.
        usage["total_tokens"] = (usage.get("input_tokens", 0)
                                 + usage.get("output_tokens", 0)
                                 + usage.get("gate_tokens", 0))
        return {
            "status": status,
            "answer": answer,
            "abstained": abstained,
            "reason": reason,
            "citations": [{"marker": c.marker, "doc_id": c.doc_id,
                           "chunk_id": c.chunk_id, "span": c.span} for c in citations],
            "retrieved": retrieved or [],
            "decisions": decisions or {},
            "usage": usage,
            "latency_ms": latency,
            "policy_version": self.policy_version,
            "policy": policy or {},
            "stages": stages or [],
            "flags": sorted(flags or []),
            "error": None,
        }


def deadlines_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)
