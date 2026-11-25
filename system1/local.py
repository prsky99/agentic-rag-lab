"""결정 모델 없이 1장 게이트를 실행해 보기 위한 로컬 대체 클라이언트.

모델 서버가 없을 때 ``python3 -m system1.demo``가 동작하도록, 질문 문구와
근거 문단의 어휘 겹침만으로 확률을 만듭니다. 정확도가 목적이 아니라 **게이트
배선과 임계값 정책이 동작하는지** 확인하는 것이 목적입니다.
"""
from __future__ import annotations

import re

TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")
STOP = {"이", "그", "저", "것", "수", "있다", "없다", "하다", "되다", "및", "그리고",
        "the", "a", "an", "of", "to", "is", "are", "and", "or", "in", "on"}
WEB_CUES = ("최신", "오늘", "현재", "뉴스", "주가", "환율", "날씨", "공시")
CHITCHAT = ("안녕", "고마워", "반가", "누구야", "뭐 해")


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(text or "") if len(t) > 1 and t not in STOP}


def overlap(a: str, b: str) -> float:
    A, B = tokens(a), tokens(b)
    if not A:
        return 0.0
    hit = 0
    for t in A:
        if t in B:
            hit += 1
        elif len(t) >= 2 and any(len(w) >= 2 and (t in w or w in t or t[:2] == w[:2])
                                 for w in B):
            hit += 1                 # 한국어 조사 차이 흡수
    return hit / len(A)


class HeuristicDecision:
    """``DecisionClient.ask``와 같은 모양의 응답을 만듭니다."""

    def ask(self, state, questions: dict) -> dict:
        out = {}
        question = str(state.get("question") or state.get("answer") or "")
        for qid, q in questions.items():
            kind = q.get("type", "choice")
            if kind == "choice":
                options = list((q.get("criteria") or {"answer": ""}).keys())
                choice, probs = self._choose(qid, question, options, state)
                out[qid] = {"type": "choice", "choice": choice,
                            "probabilities": probs, "confidence": probs[choice]}
            elif kind == "noul":
                out[qid] = {"type": "noul", "noul": self._noul(qid, state)}
            else:
                cov = self._coverage(state)
                out[qid] = {"type": "score", "score": round(4 * cov, 2)}
        return out

    # --- 선택형 ---------------------------------------------------------
    def _choose(self, qid: str, question: str, options: list[str], state: dict):
        if qid == "path":
            if any(c in question for c in CHITCHAT) or len(question.strip()) <= 4:
                pick = "no_retrieval"
            elif any(c in question for c in WEB_CUES):
                pick = "web"
            else:
                pick = "local_docs"
            if pick not in options:
                pick = options[0]
        elif qid == "strategy":
            if any(c in question for c in ("각각", "둘 다", "그리고")):
                pick = "decompose" if "decompose" in options else options[0]
            elif any(c in question for c in ("최근", "지난", "올해", "이번")):
                pick = "constrain" if "constrain" in options else options[0]
            else:
                pick = "expand" if "expand" in options else options[0]
        elif qid == "next":
            p_grounded = float(state.get("p_grounded") or 0.0)
            severity = float(state.get("severity") or 0.0)
            if p_grounded < 0.5 and any(k in question for k in
                                         ("환불", "법", "계정", "삭제", "권한", "승인")):
                pick = "human" if "human" in options else options[0]
            elif severity >= 2.0:
                pick = "bigger_model" if "bigger_model" in options else options[0]
            elif p_grounded >= 0.7:
                pick = "answer"
            else:
                pick = "abstain" if "abstain" in options else options[0]
        else:
            pick = options[0]
        probs = {o: (0.88 if o == pick else 0.12 / max(1, len(options) - 1)) for o in options}
        probs = {k: round(v, 4) for k, v in probs.items()}
        return pick, probs

    # --- 예/아니오 ------------------------------------------------------
    def _noul(self, qid: str, state: dict) -> float:
        if qid == "enough":
            return round(self._coverage(state), 4)
        if qid == "grounded":
            answer = str(state.get("answer") or "")
            spans = " ".join(str(c.get("span", "")) for c in (state.get("citations") or []))
            if not answer or not spans:
                return 0.05
            body = re.sub(r"\[\d+\]", "", answer)
            parts = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
            if not parts:
                return 0.05
            ok = sum(1 for s in parts if overlap(s, spans) >= 0.5)
            return round(min(0.95, 0.10 + 0.90 * ok / len(parts)), 4)
        if qid == "needs_fresh":
            return 0.9 if any(c in str(state.get("question", "")) for c in WEB_CUES) else 0.05
        if qid in ("multi_hop",):
            return 0.7 if "그리고" in str(state.get("question", "")) else 0.1
        return 0.5

    # --- 근거 겹침 ------------------------------------------------------
    def _coverage(self, state: dict) -> float:
        question = str(state.get("question") or "")
        passages = state.get("passages") or []
        if not passages:
            return 0.0
        text = " ".join(str(p.get("text", "")) for p in passages)
        return round(min(0.95, overlap(question, text) * 1.3), 4)
