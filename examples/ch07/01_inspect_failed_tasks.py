# examples/ch07/01_inspect_failed_tasks.py
import sys; sys.path[:0] = [".", "domains/rag"]
from domains.rag.adapter import DOMAIN as D
per, extra = D.score("runs/rag", "base", D.evolve_ids(), 2)
print([(t, r.rewards) for t, r in per.items() if min(r.rewards) < 1])
