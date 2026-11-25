# decision_head/train.py
"""출력층 문자 행 학습 (5.4절).

    python -m decision_head.train --base Qwen/Qwen3-1.7B \
        --data data/train.jsonl --out artifacts/head.pt

torch·transformers는 함수 안에서 import합니다(패키지 import만으로 GPU 의존성까지 불러오지 않도록).
"""
import argparse
import json
import random
from dataclasses import dataclass

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Example:
    prompt: str
    letter: str


def letter_token_ids(tokenizer) -> dict[str, int]:
    ids = {}
    for ch in LETTERS:
        toks = tokenizer.encode(ch, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"letter {ch!r} is not a single token: {toks}")
        ids[ch] = toks[0]
    return ids


class DecisionDataset:
    def __init__(self, path: str, tokenizer):
        self.items = []           # (prompt, gold_index, option_count)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                q = ex["question"]
                self.items.append((render_prompt(ex, tokenizer),
                                   answer_index(ex), option_count(q)))

    def __len__(self): return len(self.items)

    def __getitem__(self, i): return self.items[i]


def option_count(q: dict) -> int:
    return 2 if q["type"] == "noul" else len(q["criteria"])


def answer_index(ex: dict) -> int:
    q, a = ex["question"], ex["answer"]
    if q["type"] == "choice":
        return list(q["criteria"].keys()).index(a)
    if q["type"] == "score":
        return int(a)
    return 0 if a is True else 1      # noul: 0=true, 1=false


def render_prompt(ex: dict, tokenizer) -> str:
    q = ex["question"]
    lines = [f"State: {ex['state']}", f"Question: {q['instructions']}"]
    if q["type"] == "choice":
        for i, (name, desc) in enumerate(q["criteria"].items()):
            lines.append(f"{LETTERS[i]}. {name}: {desc or ''}")
    elif q["type"] == "score":
        for i, desc in enumerate(q["criteria"]):
            lines.append(f"{LETTERS[i]}. {desc}")
    else:
        lines.append("A. true: " + str((q.get("criteria") or {}).get("true", "")))
        lines.append("B. false: " + str((q.get("criteria") or {}).get("false", "")))
    lines.append("Answer:")
    messages = [{"role": "user", "content": "\n".join(lines)}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def load_tokenizer(base: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    # 마지막 위치의 logits를 읽으므로 패딩은 왼쪽에 둡니다(오른쪽이면 짧은 프롬프트의
    # 마지막 위치가 패딩 토큰이 됩니다).
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def train(base: str, data: str, out: str, epochs: int = 10, lr: float = 1e-3,
          batch_size: int = 16, max_len: int = 1024, eval_frac: float = 0.2,
          seed: int = 7):
    import torch
    from transformers import AutoModelForCausalLM
    random.seed(seed)
    tok = load_tokenizer(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    ids = letter_token_ids(tok)
    letters = list(ids)                 # A..Z 순서를 유지합니다(정렬하면 순서가 깨질 수 있습니다)
    rows = [ids[c] for c in letters]
    # 출력층 전체를 학습 가능으로 열고, 역전파 후 문자 행 외의 그라디언트를 0으로 만듭니다.
    # (파라미터의 일부 뷰에 requires_grad를 직접 켜면 리프 텐서가 아니라서 실패합니다.)
    head = model.get_output_embeddings()
    head.weight.requires_grad_(True)
    mask = torch.zeros(head.weight.shape[0], dtype=torch.bool, device=head.weight.device)
    mask[rows] = True
    ds = DecisionDataset(data, tok)
    idx = list(range(len(ds))); random.shuffle(idx)
    n_eval = min(max(50, int(len(ds) * eval_frac)), 1000, len(ds) // 2)
    eval_idx, train_idx = idx[:n_eval], idx[n_eval:]
    # weight_decay=0: AdamW의 가중치 감쇠는 그라디언트와 무관하게 모든 행을 줄이므로,
    # 0으로 두지 않으면 문자 행이 아닌 출력층(묶인 임베딩이면 입력 임베딩까지)이 변합니다.
    opt = torch.optim.AdamW([head.weight], lr=lr, weight_decay=0.0)
    lossf = torch.nn.CrossEntropyLoss()

    def batch_loss(batch):
        prompts = [ds.items[i][0] for i in batch]
        gold = torch.tensor([ds.items[i][1] for i in batch], device=model.device)
        ks = [ds.items[i][2] for i in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        logits = model(**enc).logits[:, -1, :]           # 마지막 위치
        cand = logits[:, rows].float()                    # 문자 행만 [B, 26]
        for j, k in enumerate(ks):                        # 이 질문에 없는 선택지는 제외
            cand[j, k:] = -1e4
        return lossf(cand, gold)

    model.train()
    for ep in range(epochs):
        random.shuffle(train_idx)
        for i in range(0, len(train_idx), batch_size):
            batch = train_idx[i:i + batch_size]
            loss = batch_loss(batch)
            opt.zero_grad(); loss.backward()
            with torch.no_grad():
                head.weight.grad[~mask] = 0.0
            opt.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(eval_idx), batch_size):
            batch = eval_idx[i:i + batch_size]
            prompts = [ds.items[j][0] for j in batch]
            ks = [ds.items[j][2] for j in batch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(model.device)
            logits = model(**enc).logits[:, -1, :]
            cand = logits[:, rows].float()
            for j, k in enumerate(ks):
                cand[j, k:] = -1e4
            pred = cand.argmax(-1).cpu()
            gold = torch.tensor([ds.items[j][1] for j in batch])
            correct += int((pred == gold).sum())
    artifact = {
        "base": base,
        "letters": letters,
        "rows": head.weight[rows].detach().float().cpu().tolist(),
        # 질문 타입별 온도(5.5절에서 홀드아웃으로 적합합니다)
        "temperature": {"choice": 1.0, "noul": 1.0, "score": 1.0},
        "eval_accuracy": correct / max(len(eval_idx), 1),
    }
    torch.save(artifact, out)
    return artifact


def main() -> int:
    ap = argparse.ArgumentParser(description="출력층 문자 행 학습 (5.4절)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, help="학습 JSONL (state/question/answer)")
    ap.add_argument("--out", default="artifacts/head.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    art = train(a.base, a.data, a.out, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size)
    print(json.dumps({"out": a.out, "eval_accuracy": round(art["eval_accuracy"], 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
