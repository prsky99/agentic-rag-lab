# 런북: 검색 품질 저하

## 신호

- `cite_precision` 하락 (승격 가드가 2포인트 하락을 막습니다)
- 기권율 상승
- `recall_by_type`에서 특정 유형만 하락

## 순서

1. **유형부터 확인합니다.** `python3 eval/recall_by_type.py --split evolve`
   - lexical만 하락 → BM25 가중·토크나이저·표기 정규화
   - semantic만 하락 → 임베딩 모델·청킹 단위
2. **청킹을 의심해 보세요.** `python3 eval/chunking/run.py`
   표나 코드 블록이 문단 경계에서 잘렸는지 확인합니다.
3. **재색인은 별칭 교체로 합니다.** 읽기를 멈추지 않습니다.
   ```bash
   python -m eval.index --tenant demo --snapshot idx-$(date +%Y%m%d)
   ```
4. **게이트 임계값은 마지막에 조정합니다.** 임계값을 낮추면 기권이 줄고 환각이
   늘어납니다. `cite_precision`과 함께 확인합니다.

## 하지 말 것

- 특정 질문이 잘 나오도록 프롬프트에 예시를 넣는 것. 그 질문만 맞힐 뿐입니다.
- 골든셋을 고쳐서 지표를 올리는 것. 감사 로그(`eval/goldenset/audit_v1.jsonl`)에
  기록이 남습니다.
