# decision_head/calibrate.py
"""질문 타입별 온도 보정 (5.5절).

    python -m decision_head.calibrate --artifact artifacts/head.pt --data data/calib.jsonl

학습에 쓰지 않은 보정용 JSONL(최소 200건 권장)로 타입별 온도 T를 적합하고,
아티팩트의 ``temperature``를 갱신해 다시 저장합니다.
"""
from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from decision_head.train import (answer_index, letter_token_ids, load_tokenizer,
                                 option_count, render_prompt)

if TYPE_CHECKING:                     # 타입 표기용(실행 시에는 읽지 않음)
    import torch


def fit_temperature(logits: "torch.Tensor", gold: "torch.Tensor",
                    lo: float = 0.5, hi: float = 10.0, steps: int = 200) -> float:
    """logits: [N, K] (문자 행), gold: [N] 인덱스. NLL을 최소화하는 T를 찾습니다."""
    import torch
    log_t = torch.zeros((), requires_grad=True)     # log(1.0)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps)
    lossf = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(lo, hi)
        loss = lossf((logits / t).float(), gold)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_t).clamp(lo, hi))


def load_model_with_head(base: str, artifact_path: str):
    """베이스를 읽고 아티팩트의 문자 행을 출력층에 덮어씁니다(server.Head와 같은 절차)."""
    import torch
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).eval()
    art = torch.load(artifact_path, map_location="cpu")
    ids = letter_token_ids(tok)
    rows = [ids[c] for c in art["letters"]]
    with torch.no_grad():
        model.get_output_embeddings().weight[rows] = torch.tensor(art["rows"]).to(model.dtype)
    return tok, model, art, rows


def collect_logits(tok, model, rows: list[int], path: str, max_len: int = 1024) -> dict:
    """질문 타입별로 (마스킹된 문자 logits [N, 26], 정답 인덱스 [N])를 모읍니다."""
    import torch
    by_type: dict[str, tuple[list, list]] = {}
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f if l.strip()]
    with torch.no_grad():
        for ex in examples:
            q = ex["question"]
            enc = tok(render_prompt(ex, tok), return_tensors="pt",
                      truncation=True, max_length=max_len).to(model.device)
            cand = model(**enc).logits[0, -1, rows].float().cpu()
            cand[option_count(q):] = -1e4                 # 학습과 같은 선택지 수 마스킹
            zs, gs = by_type.setdefault(q["type"], ([], []))
            zs.append(cand); gs.append(answer_index(ex))
    return {t: (torch.stack(zs), torch.tensor(gs)) for t, (zs, gs) in by_type.items()}


def main() -> int:
    import torch
    from decision_head.metrics import ece
    ap = argparse.ArgumentParser(description="질문 타입별 온도 보정 (5.5절)")
    ap.add_argument("--artifact", default="artifacts/head.pt")
    ap.add_argument("--data", required=True, help="학습에 쓰지 않은 보정용 JSONL")
    ap.add_argument("--base", default="", help="비우면 아티팩트의 base")
    ap.add_argument("--min-n", type=int, default=30, help="타입별 최소 예시 수")
    a = ap.parse_args()
    art0 = torch.load(a.artifact, map_location="cpu")
    tok, model, art, rows = load_model_with_head(a.base or art0["base"], a.artifact)
    report = {}
    for qtype, (z, gold) in collect_logits(tok, model, rows, a.data).items():
        if len(gold) < a.min_n:
            report[qtype] = {"n": len(gold), "skipped": "too_few"}
            continue
        t = fit_temperature(z, gold)
        before = torch.softmax(z, -1).max(-1)
        after = torch.softmax(z / t, -1).max(-1)
        correct = before.indices == gold                  # 온도를 바꿔도 순위는 그대로입니다
        art["temperature"][qtype] = t
        report[qtype] = {"n": len(gold), "T": round(t, 3),
                         "ece_before": round(ece(before.values, correct), 4),
                         "ece_after": round(ece(after.values, correct), 4)}
    torch.save(art, a.artifact)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
