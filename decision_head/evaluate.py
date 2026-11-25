# decision_head/evaluate.py
"""도입 판단용 평가 (5.7절).

    python -m decision_head.evaluate --artifact artifacts/head.pt --data data/test.jsonl

학습·보정에 쓰지 않은 평가용 JSONL에서 정확도·다수 클래스 기준선·macro-F1·ECE를 계산합니다.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from decision_head.metrics import ece


def report(preds: list[str], golds: list[str], probs: list[float],
           labels: list[str]) -> dict:
    import torch
    n = len(golds)
    acc = sum(p == g for p, g in zip(preds, golds)) / n
    majority = Counter(golds).most_common(1)[0]
    maj_acc = sum(g == majority[0] for g in golds) / n
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p != lab and g == lab)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
    conf = torch.tensor(probs); corr = torch.tensor([p == g for p, g in zip(preds, golds)])
    hi = conf >= 0.9
    return {
        "n": n, "accuracy": round(acc, 4), "majority_accuracy": round(maj_acc, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "ece": round(ece(conf, corr), 4),
        "coverage_at_0.9": round(float(hi.float().mean()), 4),
        "accuracy_at_0.9": round(float(corr[hi].float().mean()), 4) if hi.any() else None,
    }


def main() -> int:
    import torch
    from decision_head.calibrate import collect_logits, load_model_with_head
    ap = argparse.ArgumentParser(description="자체 헤드 평가 (5.7절)")
    ap.add_argument("--artifact", default="artifacts/head.pt")
    ap.add_argument("--data", required=True, help="학습·보정에 쓰지 않은 평가용 JSONL")
    ap.add_argument("--base", default="")
    a = ap.parse_args()
    art0 = torch.load(a.artifact, map_location="cpu")
    tok, model, art, rows = load_model_with_head(a.base or art0["base"], a.artifact)
    out = {}
    # 라벨은 문자 인덱스(A, B, ...)로 통일해 타입별로 채점합니다.
    for qtype, (z, gold) in collect_logits(tok, model, rows, a.data).items():
        t = art.get("temperature", {}).get(qtype, 1.0)
        p = torch.softmax(z / t, -1).max(-1)
        k = int((z > -1e3).sum(-1).max())
        labels = [chr(65 + i) for i in range(k)]
        out[qtype] = report([chr(65 + int(i)) for i in p.indices],
                            [chr(65 + int(g)) for g in gold],
                            p.values.tolist(), labels)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
