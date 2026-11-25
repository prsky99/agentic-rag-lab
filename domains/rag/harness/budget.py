"""요청 한 건의 시간·호출 예산과 공통 호출 기록 (6.5절, 10.6절).

파이프라인의 모델 호출(게이트·문서 검사·생성·재작성·재생성)은 모두 ``CallLedger``를
거칩니다. 장부는 세 가지를 합니다.

1. 호출 직전에 남은 시간과 호출 수를 검사합니다. 넘으면 ``BudgetExceeded``를 발생시키고,
   파이프라인은 그 요청을 기권으로 끝냅니다(fail-closed).
2. 남은 시간을 그 호출의 타임아웃으로 넘깁니다(``llm.call_scope``). HTTP 호출은 이 값을
   요청 전체(연결부터 본문 수신 끝까지)의 상한으로 씁니다(``llm._post_json``).
3. 호출이 끝난 직후에도 만료를 검사합니다. 호출 도중에 시간이 다 됐으면 그 결과를 쓰지 않고
   ``BudgetExceeded``를 발생시킵니다. 파이프라인은 최종 반환 직전에 한 번 더 검사하므로,
   만료된 요청은 ``answered``로 나가지 않습니다.
4. 호출마다 입력·출력 토큰을 기록합니다. 사용량은 세 상태로 구분합니다.

   * ``provider``(실측): 제공자가 돌려준 유효한 값입니다.
   * ``estimate``(추정): 원격 호출이 없는 로컬 연산(로컬 대체 판정, 추출형 생성)을 글자 수로
     어림한 값입니다.
   * ``unknown``(미측정): 원격으로 요청을 보냈지만 믿을 수 있는 사용량이 없는 호출입니다.
     실패한 호출, usage가 없거나 형식이 틀린 응답, 토큰을 세지 않는 결정 헤드의 0/0이
     여기에 듭니다. 미측정 호출은 합계에 0으로 들어가고, 글자 수 추정치는
     ``estimate_tokens``에 따로 남습니다.

시간 상한이 중단하지 못하는 작업도 있습니다. 로컬 검색, 리랭킹(교차 인코더 추론), 문맥
조립 같은 프로세스 안 연산과 이름 풀이(DNS)는 도중에 끊지 못합니다. 이런 작업은 끝난 뒤의
만료 검사로 결과만 버립니다. 상대 서버가 이미 시작한 처리도 이쪽에서 취소하지 못하므로,
생성 서비스에는 남은 시간을 ``x-deadline-ms`` 헤더로 넘겨 제공자 호출 시간을 줄이게 합니다.

예산 단위는 설정의 ``budget`` 절을 따릅니다.

* ``max_gate_calls``: G1~G4 게이트 호출 수 상한(재생성 뒤 G3 재판정 포함).
  검색 문단의 주입 검사(screen)는 게이트가 아니라 문서 검사이므로 상한에 넣지 않고
  사용량만 기록합니다.
* ``max_generation_calls``: 답변 생성 횟수 상한(최초 생성 + 재생성). 질의 재작성은
  ``gates.max_rewrites``가 따로 제한합니다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .llm import DecisionUnavailable, call_scope, estimate_tokens, validate_answers

GATE_STAGES = {"g1", "g2", "g3", "g4"}


class BudgetExceeded(RuntimeError):
    """시간 또는 호출 예산 초과. ``reason``은 deadline|gate_calls|generation_calls입니다."""

    def __init__(self, reason: str, stage: str) -> None:
        super().__init__(f"{reason} at {stage}")
        self.reason = reason
        self.stage = stage


@dataclass
class CallRecord:
    stage: str           # g1 | screen | g2 | g3 | g4 | generate | regenerate | rewrite
    kind: str            # decision | llm | extractive
    input_tokens: int
    output_tokens: int
    source: str          # provider(실측) | estimate(추정) | unknown(미측정)
    estimate_tokens: int = 0     # 미측정 호출의 글자 수 추정치(합계에는 넣지 않습니다)
    issue: str = ""              # 미측정 사유: failed | missing | invalid | not_counted

    def as_dict(self) -> dict:
        out = asdict(self)
        if self.source != "unknown":             # 실측·추정 기록의 모양은 그대로 둡니다
            out.pop("estimate_tokens")
            out.pop("issue")
        return out


class CallLedger:
    def __init__(self, deadline, max_gate_calls: int, max_generation_calls: int) -> None:
        self.deadline = deadline
        self.max_gate_calls = int(max_gate_calls)
        self.max_generation_calls = int(max_generation_calls)
        self.gate_calls = 0
        self.generation_calls = 0
        self.calls: list[CallRecord] = []
        self.flags: set[str] = set()             # 응답의 flags에 합칩니다(usage_invalid@단계)

    # --- 예산 검사 -------------------------------------------------------
    def check_deadline(self, stage: str) -> None:
        if self.deadline.expired():
            raise BudgetExceeded("deadline", stage)

    def begin(self, stage: str, generation: bool = False) -> None:
        """호출 직전 검사. 통과하면 호출 수를 하나 올립니다."""
        self.check_deadline(stage)
        if stage in GATE_STAGES:
            if self.gate_calls >= self.max_gate_calls:
                raise BudgetExceeded("gate_calls", stage)
            self.gate_calls += 1
        if generation:
            if self.generation_calls >= self.max_generation_calls:
                raise BudgetExceeded("generation_calls", stage)
            self.generation_calls += 1

    def timeout_s(self) -> float:
        return max(0.001, self.deadline.left_ms() / 1000.0)

    # --- 기록 ------------------------------------------------------------
    def record(self, stage: str, kind: str, input_tokens: int, output_tokens: int,
               source: str) -> None:
        self.calls.append(CallRecord(stage, kind, int(input_tokens), int(output_tokens),
                                     source))

    def record_unknown(self, stage: str, kind: str, estimate: int, issue: str) -> None:
        """미측정 호출을 남깁니다. 토큰은 0으로 두고 추정치와 사유를 따로 적습니다."""
        self.calls.append(CallRecord(stage, kind, 0, 0, "unknown",
                                     estimate_tokens=max(0, int(estimate)),
                                     issue=issue or "missing"))
        if issue == "invalid":
            self.flags.add(f"usage_invalid@{stage}")

    def usage(self) -> dict:
        llm_in = sum(c.input_tokens for c in self.calls if c.kind != "decision")
        llm_out = sum(c.output_tokens for c in self.calls if c.kind != "decision")
        gate = sum(c.input_tokens + c.output_tokens for c in self.calls if c.kind == "decision")
        provider = sum(c.input_tokens + c.output_tokens for c in self.calls
                       if c.source == "provider")
        estimated = sum(c.input_tokens + c.output_tokens for c in self.calls
                        if c.source != "provider")
        sources = {c.source for c in self.calls}
        unknown = [c for c in self.calls if c.source == "unknown"]
        out = {
            "input_tokens": llm_in,
            "output_tokens": llm_out,
            "gate_tokens": gate,
            "total_tokens": llm_in + llm_out + gate,
            "provider_tokens": provider,
            "estimated_tokens": estimated,
            "source": (sources.pop() if len(sources) == 1 else
                       ("mixed" if sources else "none")),
            # 미측정 호출이 하나도 없을 때만 true입니다. false이면 total_tokens는 최솟값입니다.
            "usage_known": not unknown,
            "unknown_calls": len(unknown),
            "gate_calls": self.gate_calls,
            "generation_calls": self.generation_calls,
            "calls": [c.as_dict() for c in self.calls],
        }
        if unknown:
            out["total_is_minimum"] = True
            # 미측정 호출의 글자 수 추정치 합계. 평가의 비용 C는 이 값을 더해 씁니다.
            out["unknown_estimate_tokens"] = sum(c.estimate_tokens for c in unknown)
        return out


class MeteredDecision:
    """결정 모델 클라이언트를 감싸 예산 검사·타임아웃 전달·응답 검증·사용량 기록을 합니다.

    게이트 함수는 ``client.ask(state, questions)``만 부르므로, 단계 이름을 붙인 이 객체를
    그대로 넘기면 됩니다.
    """

    def __init__(self, client, ledger: CallLedger, stage: str) -> None:
        self.client = client
        self.ledger = ledger
        self.stage = stage

    def ask(self, state, questions: dict) -> dict:
        self.ledger.begin(self.stage)
        in_est = estimate_tokens(json.dumps(state, ensure_ascii=False)) \
            + estimate_tokens(json.dumps(questions, ensure_ascii=False))
        out_est = len(questions)                # 질문당 판단 값 1개
        remote = getattr(self.client, "mode", "") == "remote"
        scope: dict = {"usage": None}
        ok = False
        try:
            with call_scope(self.ledger.timeout_s()) as scope:
                answers = self.client.ask(state, questions)
            validate_answers(questions, answers)
            ok = True
        except DecisionUnavailable as exc:
            # 시간이 다 돼서 끊긴 호출은 결정 모델 장애가 아니라 시간 초과로 끝냅니다.
            if self.ledger.deadline.expired():
                raise BudgetExceeded("deadline", self.stage) from exc
            raise
        finally:
            # 실패한 호출도 기록합니다. 원격 호출의 사용량은 제공자가 준 유효한 값만 실측으로
            # 받고, 그 밖에는 미측정으로 남깁니다. 로컬 대체 판정은 추정치로 남깁니다.
            u = scope.get("usage")
            issue = scope.get("usage_issue") or ""
            if u:
                self.ledger.record(self.stage, "decision", u.get("input_tokens", 0),
                                   u.get("output_tokens", 0), "provider")
            elif remote or issue:
                self.ledger.record_unknown(self.stage, "decision", in_est + out_est,
                                           issue or ("missing" if ok else "failed"))
            else:
                self.ledger.record(self.stage, "decision", in_est, out_est, "estimate")
        # 호출 도중에 시간이 다 됐으면 그 판정으로 다음 단계를 진행하지 않습니다(RT04).
        self.ledger.check_deadline(self.stage)
        return answers
