"""트라이얼 결과 분류표 — 기권 사유를 정상 기권·보안 차단·운영 장애·예산 초과로 나눕니다(R01).

진화 채점(adapter.py), 회귀 평가(eval/run_subset.py), 임계값 검사(eval/check_thresholds.py)가
모두 이 표 하나를 씁니다. 외부 패키지를 import하지 않으므로 임계값 검사에서도 가볍게 불러옵니다.

파이프라인(harness/pipeline.py)은 기권할 때 ``abstained=true``와 함께 ``reason``을 남깁니다.
그런데 같은 기권이라도 뜻이 다릅니다. 근거가 모자라 멈춘 기권은 답변 불가 문항에서 정답 행동이지만,
시간이 다 되었거나 검색기가 죽어서 멈춘 기권은 검토 자체를 하지 못한 실패입니다. 그래서 사유마다
결과 종류를 정하고, 표에 없는 사유는 ``unknown``(실패)으로 봅니다. 새 사유가 생겼는데 표를 고치지
않으면 만점이 아니라 0점이 되도록 보수적으로 처리합니다.
"""
from __future__ import annotations

# 결과 종류
ANSWERED = "answered"            # 답변
ABSTAINED = "abstained"          # 정상 기권: 근거 부족·정책에 따른 기권
BLOCKED = "blocked"              # 보안 차단: 입력·출력 가드
OPS_FAILURE = "ops_failure"      # 운영 장애: 결정·검색·생성 구성 요소 장애
BUDGET = "budget_exceeded"       # 예산 초과: 데드라인·호출 상한 소진
UNKNOWN = "unknown"              # 표에 없는 기권 사유(보수적으로 실패)
ERROR = "error"                  # 실행 오류(error 필드 있음)

OUTCOMES = (ANSWERED, ABSTAINED, BLOCKED, OPS_FAILURE, BUDGET, UNKNOWN, ERROR)

# 기권으로 인정하는 결과. 답변 불가 문항에서 이 둘은 1점입니다.
# 보안 차단 정책: 답변 불가 문항에서 가드가 막은 경우는 사용자에게 아무 답도 나가지 않았다는 점에서
# 정상 기권과 결과가 같으므로 기권으로 인정합니다. 답변 가능 문항에서 막힌 경우는 답을 주지 못했으므로
# 오답(정답 0·답변 0)이고, 답변 가능 문항의 기권율(과잉 기권)에 들어갑니다.
REFUSALS = frozenset({ABSTAINED, BLOCKED})

# 실패로 세는 결과. 기권이어도 0점이고 error_rate(실패율)에 합산합니다.
FAILURES = frozenset({OPS_FAILURE, BUDGET, UNKNOWN, ERROR})

# 파이프라인이 실제로 내는 기권 사유 전체(pipeline.py의 _finish 호출, gates.should_retry의 why).
ABSTAIN_REASONS: dict[str, str] = {
    # 정상 기권 — 근거 부족·정책
    "insufficient_evidence": ABSTAINED,     # G2가 근거 부족으로 판정
    "stop_max_attempts": ABSTAINED,         # 재검색 상한까지 했지만 근거 부족
    "stop_no_gain": ABSTAINED,              # 재검색해도 충분도가 오르지 않음
    "stop_same_evidence": ABSTAINED,        # 재검색해도 같은 근거
    "ungrounded": ABSTAINED,                # G3 불일치 후 G4가 기권 결정
    "human_required": ABSTAINED,            # G4가 사람에게 넘김(status=escalated)
    "web_unsupported": ABSTAINED,           # web_policy=abstain 정책 기권
    # 보안 차단 — 입력·출력 가드
    "injection_in_question": BLOCKED,
    "output_blocked_prompt_leak": BLOCKED,
    "output_blocked_secret_pattern": BLOCKED,
    # 운영 장애 — 결정·검색·생성 구성 요소
    "retrieval_unavailable": OPS_FAILURE,
    "g3_unavailable": OPS_FAILURE,
    # 예산 초과 — 시간·호출 예산
    "deadline_exceeded": BUDGET,
    "call_budget_exceeded": BUDGET,
    "stop_deadline": BUDGET,                # 재검색 직전 남은 시간 부족
}


def outcome(tr: dict) -> str:
    """트라이얼 결과의 종류(``OUTCOMES`` 중 하나).

    ``error``를 먼저 봅니다. 파이프라인은 검색 실패처럼 실행이 깨진 경우에도 ``abstained=true``를
    돌려주므로(통제된 기권), 기권 여부보다 오류 여부가 앞섭니다(E05). 기권이면 사유 표로 나누고,
    표에 없는 사유는 ``unknown``입니다(R01).
    """
    if tr.get("error"):
        return ERROR
    if not tr.get("abstained"):
        return ANSWERED
    return ABSTAIN_REASONS.get(str(tr.get("reason") or ""), UNKNOWN)
