# 훈련: 결정 모델 다운 (30분)

## 목표

G3가 죽었을 때 파이프라인이 **기권**하는지, G1이 죽었을 때 **검색으로 폴백**하는지
직접 확인해 보세요.

## 준비

```bash
make up
make token
```

## 실행

```bash
# 1. 정상 기준선
bash tests/verify.sh

# 2. 결정 모델만 내립니다(복구용으로 현재 replica 수를 먼저 기록합니다)
PREV=$(kubectl -n rag get deploy/decision -o jsonpath='{.spec.replicas}')
trap 'kubectl -n rag scale deploy/decision --replicas="${PREV:-1}"' EXIT   # 중간에 끊겨도 복구
kubectl -n rag scale deploy/decision --replicas=0
# 중단 확인: "중단 확인 실패"가 찍히면(120초 안에 파드가 남아 있으면) 3단계로 넘어가지 않습니다.
kubectl -n rag wait --for=delete pod -l app=decision --timeout=120s \
  && echo "중단 확인" || echo "중단 확인 실패 - 3단계로 넘어가지 마세요"
kubectl -n rag get pods -l app=decision          # "No resources found"가 나와야 합니다

# 3. 같은 질의 — 기권 사유까지 확인합니다(사유가 없거나 다르면 jq -e가 실패합니다)
curl -s http://rag.localtest.me/v1/answer \
  -H "Authorization: Bearer $(cat /tmp/rag-token)" \
  -H 'content-type: application/json' \
  -d '{"question":"환불은 며칠 이내에 가능한가요?"}' \
  | tee /dev/stderr \
  | jq -e '.abstained == true and .status == "abstained" and .reason == "g3_unavailable"'

# 4. 복구
kubectl -n rag scale deploy/decision --replicas="${PREV:-1}"   # 2단계에서 기록한 값
kubectl -n rag rollout status deploy/decision --timeout=180s
trap - EXIT
```

## 기대

- 2단계: decision 파드 0개(중단을 확인하지 못했다면 훈련 결과로 인정하지 않습니다)
- 3단계: `abstained=true`, `status=abstained`, `reason=g3_unavailable` — 마지막 줄에 `true`가 찍힙니다.
  사유가 비어 있거나 다른 값이면 실패입니다(예: `insufficient_evidence`는 G2 폴백이 먼저 기권한 경우로,
  G3 fail-closed를 확인한 것이 아닙니다)
- HTTP는 200 (게이트웨이는 살아 있습니다). 응답까지 데드라인만큼(약 6초) 걸립니다
- 같은 조건을 자동으로 확인하는 것이 `tests/verify.sh`의 마지막 시나리오입니다
- 복구 후 `make verify` 전부 통과

## 토의

- G3를 fail-open으로 바꾸면 어떤 일이 생길까요?
- 기권 문구가 사용자에게 충분할까요?
