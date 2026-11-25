# examples/ch11/03_filter_by_trust.py
def filter_by_trust(chunks, min_trust="internal"):
    rank = {"trusted": 2, "internal": 1, "untrusted": 0}
    return [c for c in chunks if rank[c.trust] >= rank[min_trust]]
