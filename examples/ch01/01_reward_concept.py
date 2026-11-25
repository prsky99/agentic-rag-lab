# examples/ch01/01_reward_concept.py
# 개념 코드 — 실제 구현은 domains/rag/judge.py의 answer_reward (8.4절)
def score_trial(t: Trial, baseline: Trial) -> float:
    correct      = 1.0 if t.answer_matches_gold else 0.0
    cite_prec    = t.cited_supported / max(t.cited_total, 1)      # 인용 정밀도
    abstain_ok   = 1.0 if (t.should_abstain and t.abstained) else 0.0
    hallucinated = 1.0 if t.answered_when_should_abstain else 0.0
    s = 0.60 * correct + 0.20 * cite_prec + 0.20 * abstain_ok
    s -= 0.30 * hallucinated          # 기권했어야 할 때 답하면 강하게 감점
    return s
def cost_ratio(t: Trial, baseline: Trial) -> float:
    return t.policy_tokens / max(baseline.policy_tokens, 1)
