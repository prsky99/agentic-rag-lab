# 런북: 결정 모델 다운

## 증상

- `g1_fallback` 로그가 급증
- 기권율 상승, p95 변화 없음
- orchestrator 로그에 `DecisionUnavailable`

## 즉시 확인

```bash
kubectl -n rag get pods -l app=decision
kubectl -n rag logs deploy/decision --tail=100
curl -s -o /dev/null -w "%{http_code}\n" http://rag.localtest.me/v1/answer \
  -H "Authorization: Bearer $(cat /tmp/rag-token)" \
  -H 'content-type: application/json' -d '{"question":"환불 규정"}'
```

## 정책

| 게이트 | 다운 시 | 이유 |
|---|---|---|
| G1 라우터 | `local_docs`로 폴백 | 가용성. 잘못 라우팅해도 검색·G2가 막습니다 |
| G2 충분성 | 리랭커 점수 기반 보수 판정 | 기권 쪽으로 떨어집니다 |
| G3 충실성 | **기권** (fail-closed) | 검증 없는 답변을 내보내지 않습니다 |

## 복구

```bash
kubectl -n rag rollout restart deploy/decision
kubectl -n rag rollout status deploy/decision --timeout=180s
```

복구 후 `make verify`로 다섯 시나리오를 다시 확인해 주세요.
