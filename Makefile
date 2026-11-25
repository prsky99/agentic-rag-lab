CLUSTER := rag-lab
NS := rag
TENANT ?= demo
T ?= 1
SNAPSHOT ?= idx-2026-11-20
BASE ?= http://rag.localtest.me
IMAGES := gateway orchestrator retriever reranker generator guard decision
OVERLAY ?= deploy/overlays/kind
# Qdrant 백엔드용 오버레이: $(OVERLAY)에 deploy/components/qdrant-backend(모델 캐시 볼륨,
# startupProbe, 모델 내려받기 egress)를 더한 것입니다. deploy/base/rag-config.env의
# RETRIEVER_BACKEND가 qdrant일 때만 적용하고, 기본값(local)이면 $(OVERLAY)를 그대로 적용합니다.
QDRANT_OVERLAY ?= $(OVERLAY)-qdrant
APPLY_DIR = $(if $(shell grep -x 'RETRIEVER_BACKEND=qdrant' deploy/base/rag-config.env),$(QDRANT_OVERLAY),$(OVERLAY))
# 설정(rag-config)을 환경 변수로 읽는 Deployment
CONFIG_DEPLOYS := gateway orchestrator retriever reranker generator guard decision
KEY ?= RETRIEVER_BACKEND
ALIAS ?= docs_current
QDRANT_LOCAL_PORT ?= 16333
CHECK_TENANT ?= acme
EXPECT ?= qdrant
# 4장 결정 서비스의 shim 이미지(직접 빌드)
SHIM_REGISTRY ?= registry.example.com/rag
SHIM_TAG ?= 0.1.0

.PHONY: up down image deploy token index smoke load eval eval-baseline \
        evolve-baseline evolve-run ood verify verify-local test goldenset audit \
        config config-show config-prune use-qdrant backend-check shim-image

# ---------------------------------------------------------------- 클러스터
up: image
	kind get clusters | grep -qx $(CLUSTER) || kind create cluster --config deploy/kind/kind-config.yaml
	kind load docker-image $(addprefix rag/,$(addsuffix :dev,$(IMAGES))) --name $(CLUSTER)
	kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.15.1/deploy/static/provider/kind/deploy.yaml
	kubectl -n ingress-nginx patch deploy ingress-nginx-controller --type=merge \
	  -p '{"spec":{"template":{"spec":{"nodeSelector":{"ingress-ready":"true"}}}}}'
	kubectl -n ingress-nginx rollout status deploy/ingress-nginx-controller --timeout=180s
	kubectl get ns $(NS) >/dev/null 2>&1 || kubectl create namespace $(NS)
	@if [ -f deploy/kind/01-secret.yaml ]; then kubectl apply -f deploy/kind/01-secret.yaml; \
	else kubectl -n $(NS) create secret generic rag-secrets \
	  --from-literal=JWT_SECRET=dev-only-secret \
	  --from-literal=INTERNAL_TOKEN=dev-internal-token \
	  --dry-run=client -o yaml | kubectl apply -f -; fi
	$(MAKE) deploy
	kubectl -n $(NS) wait --for=condition=Available deploy --all --timeout=300s
	kubectl -n $(NS) rollout status statefulset/qdrant --timeout=180s

image:
	@for svc in $(IMAGES); do \
	  docker build -t rag/$$svc:dev -f services/$$svc/Dockerfile . || exit 1; \
	done

deploy:
	kubectl apply -k $(APPLY_DIR)

# ---------------------------------------------------------------- 설정 반영
# rag-config는 해시 이름 ConfigMap(rag-config-<해시>)입니다. deploy/base/rag-config.env를 고치고
# 적용하면 이름이 바뀌어 관련 Deployment가 롤링 재시작됩니다. 이 타깃은 적용 → 재시작 완료 대기 →
# 파드가 실제로 읽은 값 출력까지 한 번에 합니다. (GKE: make config OVERLAY=deploy/overlays/gke)
config:
	kubectl apply -k $(APPLY_DIR)
	@for d in $(CONFIG_DEPLOYS); do \
	  kubectl -n $(NS) rollout status deploy/$$d --timeout=600s || exit 1; \
	done
	@$(MAKE) --no-print-directory config-show

# 파드가 실제로 읽은 값 확인: make config-show KEY=RETRIEVER_BACKEND
config-show:
	@echo "retriever가 참조하는 ConfigMap: $$(kubectl -n $(NS) get deploy/retriever \
	  -o jsonpath='{.spec.template.spec.containers[0].envFrom[0].configMapRef.name}')"
	@for d in $(CONFIG_DEPLOYS); do \
	  printf '%-13s %s=%s\n' "$$d" "$(KEY)" \
	    "$$(kubectl -n $(NS) exec deploy/$$d -- printenv $(KEY) 2>/dev/null || echo '<없음>')"; \
	done

# 더는 쓰지 않는 예전 rag-config-<해시>를 지웁니다. 롤링 재시작이 끝난 뒤에만 실행합니다.
# 지우면 `kubectl rollout undo`로 예전 설정에 돌아갈 수 없으므로 되돌릴 일이 없을 때 정리합니다.
config-prune:
	@cur=$$(kubectl -n $(NS) get deploy/retriever \
	  -o jsonpath='{.spec.template.spec.containers[0].envFrom[0].configMapRef.name}'); \
	[ -n "$$cur" ] || { echo "[FAIL] 현재 ConfigMap 이름을 찾지 못했습니다"; exit 1; }; \
	for cm in $$(kubectl -n $(NS) get configmap -o name | grep '^configmap/rag-config'); do \
	  [ "$$cm" = "configmap/$$cur" ] || kubectl -n $(NS) delete "$$cm"; \
	done

# Qdrant 백엔드로 전환: 색인 성공 확인 → RETRIEVER_BACKEND=qdrant 적용 → retriever 재시작 대기
# → 실제 backend 확인. 색인이 비어 있으면 1단계에서 멈추고 설정을 바꾸지 않습니다.
# 2단계는 Qdrant용 오버레이($(QDRANT_OVERLAY))를 적용합니다. retriever 파드는 뜰 때마다 임베딩 모델
# (기본 약 0.22GB)을 내려받아 읽은 뒤에야 /ready를 통과하므로, 3단계 대기 시간을 넉넉히 둡니다
# (startupProbe가 파드 하나에 최대 5분을 기다립니다).
use-qdrant:
	@echo "[1/4] 색인 확인: Qdrant 별칭 $(ALIAS)의 점 개수"
	@kubectl -n $(NS) port-forward svc/qdrant $(QDRANT_LOCAL_PORT):6333 >/dev/null 2>&1 & pf=$$!; \
	trap "kill $$pf 2>/dev/null" EXIT; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
	  curl -fsS -o /dev/null http://127.0.0.1:$(QDRANT_LOCAL_PORT)/readyz 2>/dev/null && break; sleep 1; \
	done; \
	n=$$(curl -fsS -X POST -H 'content-type: application/json' -d '{"exact":true}' \
	  http://127.0.0.1:$(QDRANT_LOCAL_PORT)/collections/$(ALIAS)/points/count | jq -r '.result.count'); \
	case "$$n" in ''|null|0|*[!0-9]*) \
	  echo "[FAIL] 별칭 $(ALIAS)에 점이 없습니다(받은 값: '$$n'). 먼저 색인해 주세요:"; \
	  echo "       kubectl -n $(NS) port-forward svc/qdrant 6333:6333 &"; \
	  echo "       QDRANT_URL=http://localhost:6333 make index"; exit 1;; \
	esac; echo "       점 $$n개"
	@echo "[2/4] RETRIEVER_BACKEND=qdrant 적용 (deploy/base/rag-config.env 수정 → $(QDRANT_OVERLAY) apply)"
	sed -i.bak 's/^RETRIEVER_BACKEND=.*/RETRIEVER_BACKEND=qdrant/' deploy/base/rag-config.env
	@rm -f deploy/base/rag-config.env.bak
	kubectl apply -k $(QDRANT_OVERLAY)
	@echo "[3/4] retriever 롤링 재시작 대기 (모델 내려받기 포함)"
	kubectl -n $(NS) rollout status deploy/retriever --timeout=600s
	@echo "[4/4] 실제 backend 확인"
	@$(MAKE) --no-print-directory backend-check EXPECT=qdrant

# retriever 파드가 읽은 환경 변수와 /search 응답의 backend 필드를 함께 봅니다.
# (/healthz에는 backend 필드가 없어 파드 안에서 /search를 한 번 부릅니다.)
backend-check:
	@env=$$(kubectl -n $(NS) exec deploy/retriever -- printenv RETRIEVER_BACKEND); \
	got=$$(kubectl -n $(NS) exec deploy/retriever -- python -c "import json,urllib.request as u; r=u.Request('http://127.0.0.1:8080/search', data=json.dumps({'query':'refund','tenant':'$(CHECK_TENANT)','top_k':1}).encode(), headers={'content-type':'application/json'}); d=json.load(u.urlopen(r, timeout=20)); print(d['backend'])"); \
	echo "환경 변수 RETRIEVER_BACKEND=$$env, /search 응답 backend=$$got (기대: $(EXPECT))"; \
	[ "$$env" = "$(EXPECT)" ] && [ "$$got" = "$(EXPECT)" ] || { echo "[FAIL] backend가 $(EXPECT)가 아닙니다"; exit 1; }; \
	echo "[PASS] retriever backend=$$got"

# ---------------------------------------------------------------- 4장 shim 이미지
# deploy/decision/10-decision.yaml의 shim 컨테이너 이미지입니다(공개 이미지 없음, 직접 빌드).
#   make shim-image SHIM_REGISTRY=asia-northeast3-docker.pkg.dev/PROJECT/rag
#   docker push <SHIM_REGISTRY>/openjev-shim:$(SHIM_TAG)
# 모델 커밋을 바꾸려면 SHIM_BUILD_ARGS="--build-arg MODEL_REVISION=<해시> --build-arg SHIM_SHA256=<해시>"
shim-image:
	docker build -t $(SHIM_REGISTRY)/openjev-shim:$(SHIM_TAG) $(SHIM_BUILD_ARGS) \
	  -f deploy/decision/shim/Dockerfile deploy/decision/shim

down:
	kind delete cluster --name $(CLUSTER)

# ---------------------------------------------------------------- 로컬 실행
token:
	python3 tools/dev_token.py --tenant $(TENANT) --tier pro > /tmp/rag-token
	python3 tools/dev_token.py --tenant other --tier pro > /tmp/rag-token-b

index:
	python3 -m eval.index --corpus domains/rag/data/corpus --tenant $(TENANT) \
	  --alias docs_current --snapshot $(SNAPSHOT)

smoke:
	curl -fsS -H "Authorization: Bearer $$(cat /tmp/rag-token)" \
	  -H "content-type: application/json" \
	  -d '{"question":"환불은 며칠 이내에 가능한가요?"}' \
	  $(BASE)/v1/answer | jq .

load:
	k6 run load/rag.js

# ---------------------------------------------------------------- 평가
eval:
	python3 eval/run_subset.py --subset eval/ci/regression_subset.json --out eval/reports/kind.json

eval-baseline:
	python3 eval/run_subset.py --subset eval/ci/regression_subset.json --out /tmp/base_candidate.json
	python3 eval/check_thresholds.py /tmp/base_candidate.json --update-baseline

goldenset:
	python3 tools/build_goldenset.py

audit:
	python3 eval/audit_goldenset.py

# ---------------------------------------------------------------- 진화 루프
evolve-baseline:
	python3 rrsi.py --domain rag baseline --job base1
	python3 rrsi.py --domain rag baseline --job base2
	python3 rrsi.py --domain rag baseline --job base3
	python3 rrsi.py --domain rag calibrate --jobs base1,base2,base3

evolve-run:
	python3 rrsi.py --domain rag --T $(T) run

ood:
	RAG_HELDOUT_SPLIT=ood_docs python3 rrsi.py --domain rag heldout --label ood_docs
	RAG_HELDOUT_SPLIT=ood_intent python3 rrsi.py --domain rag heldout --label ood_intent

# ---------------------------------------------------------------- 검증
verify:
	bash tests/verify.sh

verify-local:
	python3 tests/local_verify.py

test:
	python3 -m pytest tests -q
	python3 -m compileall -q $(wildcard domains services rrsi eval system1 tools)
