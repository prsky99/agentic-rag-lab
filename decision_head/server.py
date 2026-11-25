# decision_head/server.py
"""자체 헤드를 ``/v1/systemone`` 계약으로 서빙합니다 (5.6절).

    DECISION_HEAD_BASE=Qwen/Qwen3-1.7B DECISION_HEAD_ARTIFACT=artifacts/head.pt \
        uvicorn decision_head.server:app --port 3000

모델은 첫 요청 때 읽습니다(import만으로 torch·transformers까지 불러오지 않도록).
"""
import hashlib
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Question(BaseModel):
    type: str
    instructions: str | dict | list
    criteria: dict | list | None = None


class Request(BaseModel):
    model: str = "local-head"
    state: str | dict | list
    questions: dict[str, Question]


app = FastAPI()


def render(state, q: Question, tok) -> str:
    lines = [f"State: {state}", f"Question: {q.instructions}"]
    if q.type == "choice" and isinstance(q.criteria, dict):
        for i, (name, desc) in enumerate(q.criteria.items()):
            lines.append(f"{LETTERS[i]}. {name}: {desc or ''}")
    elif q.type == "score" and isinstance(q.criteria, list):
        for i, desc in enumerate(q.criteria):
            lines.append(f"{LETTERS[i]}. {desc}")
    else:
        crit = q.criteria if isinstance(q.criteria, dict) else {}
        lines.append("A. true: " + str(crit.get("true", "")))
        lines.append("B. false: " + str(crit.get("false", "")))
    lines.append("Answer:")
    messages = [{"role": "user", "content": "\n".join(lines)}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class Head:
    def __init__(self, base: str, artifact: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base)
        self.m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).eval()
        art = torch.load(artifact, map_location="cpu")
        rows = [self.tok.encode(c, add_special_tokens=False)[0] for c in art["letters"]]
        self.ids = dict(zip(art["letters"], rows))
        with torch.no_grad():
            self.m.get_output_embeddings().weight[rows] = (
                torch.tensor(art["rows"]).to(self.m.dtype))
        self.temp = {q: art.get("temperature", {}).get(q, 1.0)
                     for q in ("choice", "noul", "score")}
        digest = hashlib.sha256(
            torch.tensor(art["rows"], dtype=torch.float32).numpy().tobytes()).hexdigest()[:12]
        self.version = f"{base} head={digest} T={self.temp}"

    def predict(self, state, q: Question, k: int) -> dict:
        import torch
        prompt = render(state, q, self.tok)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            logits = self.m(**enc).logits[0, -1]
        letters = list(self.ids)[:k]          # 이 질문에 실제로 있는 선택지만
        z = torch.tensor([float(logits[self.ids[c]]) for c in letters])
        t = self.temp[q.type]
        p = torch.softmax(z / t, dim=-1)
        return {"letters": letters, "p": p.tolist()}


BASE = os.environ.get("DECISION_HEAD_BASE", "Qwen/Qwen3-1.7B")  # 상업 이용이 허용되는 베이스로 교체해 주세요
ARTIFACT = os.environ.get("DECISION_HEAD_ARTIFACT", "artifacts/head.pt")
_head: Head | None = None


def get_head() -> Head:
    global _head
    if _head is None:
        _head = Head(BASE, ARTIFACT)
    return _head


@app.post("/v1/systemone")
def systemone(req: Request):
    head = get_head()
    t0 = time.perf_counter()
    answers = {}
    for qid, q in req.questions.items():
        k = 2 if q.type == "noul" else len(q.criteria or [])
        if not 2 <= k <= len(LETTERS):
            raise HTTPException(422, f"{qid}: options must be 2..{len(LETTERS)}, got {k}")
        out = head.predict(req.state, q, k)
        p = out["p"]; letters = out["letters"]
        if q.type == "choice":
            names = list(q.criteria.keys())
            probs = {n: round(p[i], 4) for i, n in enumerate(names)}
            best = max(probs, key=probs.get)
            top = probs[best]
            n = len(names)
            # 균등 분포(1/n)를 0, 확신(1.0)을 1로 옮긴 정규화 신뢰도
            answers[qid] = {"type": "choice", "choice": best, "probabilities": probs,
                            "confidence": round((top - 1 / n) / (1 - 1 / n), 4)}
        elif q.type == "noul":
            answers[qid] = {"type": "noul", "noul": round(p[letters.index("A")], 4)}
        elif q.type == "score":
            levels = q.criteria
            probs = {str(i): round(p[i], 4) for i in range(len(levels))}
            score = sum(i * p[i] for i in range(len(levels)))
            answers[qid] = {"type": "score", "score": round(float(score), 3),
                            "legend": {str(i): d for i, d in enumerate(levels)},
                            "probabilities": probs, "confidence": round(max(p), 4)}
        else:
            raise HTTPException(422, f"unknown type {q.type}")
    return {"id": f"head-{int(time.time()*1000)}", "model": head.version,
            "answers": answers,
            # 이 헤드는 토큰을 세지 않습니다. 0/0은 실측이 아니므로 usage_known=false를
            # 함께 싣고, 클라이언트는 이 호출을 미측정으로 기록합니다.
            "usage": {"input_tokens": 0, "output_tokens": 0}, "usage_known": False,
            "latency_ms": int((time.perf_counter() - t0) * 1000)}


@app.get("/v1/version")
def version():
    return {"model": get_head().version, "base": BASE}
