# eval/goldenset

- `split_rag.json` — 분할 목록. `domains/rag/data/split_rag.json`의 사본이며,
  `python3 tools/build_goldenset.py`가 둘을 같은 내용으로 다시 씁니다.
- `audit_v1.jsonl` — `python3 eval/audit_goldenset.py`의 결과. 문제가 없으면 빈 파일입니다.
  저장소에는 커밋하지 않고, CI에서 매번 새로 만듭니다.
- 골든셋 본문은 `domains/rag/data/goldenset_v1.jsonl` 하나만 둡니다(사본 금지).
