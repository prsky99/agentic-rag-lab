# 런북: 승격 되돌리기

## 언제

승격 직후 지표가 나빠졌거나, 고객 영향이 확인됐을 때. 판단은 5분 안에 내립니다.
원인 분석은 롤백 뒤에 합니다.

## 절차

```bash
# 직전 승격으로 되돌리기
git checkout evolve/rag
git reset --hard HEAD~1        # 승격 커밋이 하나라면
git push --force-with-lease origin evolve/rag

# 같은 하네스로 이미지를 다시 굽고 롤아웃
HARNESS=$(git rev-parse --short evolve/rag)
docker build -t rag/orchestrator:harness-$HARNESS -f services/orchestrator/Dockerfile .
kubectl -n rag set image deploy/orchestrator orchestrator=rag/orchestrator:harness-$HARNESS
kubectl -n rag rollout status deploy/orchestrator --timeout=180s
```

## 확인

```bash
make verify                     # 다섯 시나리오
kubectl -n rag logs deploy/orchestrator --tail=100 | grep -c DecisionUnavailable
```

## 기록

- 되돌린 커밋 해시, 되돌린 이유(지표 이름과 값), 영향 시간
- `runs/rag/history.jsonl`에서 해당 라운드의 `reason`을 확인합니다
