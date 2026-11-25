# agentic-rag-lab

이 저장소는 전자책 **『클라우드 네이티브 쿠버네티스 기반 차세대 Agentic RAG 구축』**
(한태석)의 실습 저장소입니다. 1장의 게이트 4종에서 12장의 엔드투엔드 실습까지, 본문에
나오는 코드와 매니페스트·시험을 이 저장소 하나에 모았습니다. 본문 코드 블록 첫 줄의
`# 경로` 주석이 이 저장소 안의 파일 경로입니다.

저장소 받기(둘 중 하나):

```bash
git clone https://github.com/prsky99/agentic-rag-lab.git    # 1) 복제 — git 작업 트리
cd agentic-rag-lab
# 2) 또는 저장소 화면의 Code → Download ZIP으로 내려받아 풉니다
```

ZIP으로 받은 폴더는 git 작업 트리가 아니므로, 진화 실행(`rrsi.py ... run`)이 후보마다
`git worktree` 대신 폴더 복사를 쓰는 복사 모드(`workspace=copy`)로 동작합니다(본문 7.5절·부록 A.7).
승격을 커밋으로 남기려면 `git init && git add -A && git commit -m init`을 먼저 실행해 주세요.

- 하네스(G1→검색→G2→생성→G3→G4)가 **실제로 돈다**: kind 없이도 로컬 모드로 질의 1건이
  끝까지 흐릅니다.
- 평가와 프로덕션이 **같은 코드**를 씁니다. `domains/rag/harness`가 유일한 편집면입니다.
- RRSI 루프(분석→제안→심사→평가→선택→승격)가 `runs/rag/`에 이력과 frontier를 남깁니다.
- 골든셋·감사·회귀 임계값·테넌트 격리·인젝션·장애 주입 테스트가 들어 있습니다.

> **먼저 알아 둘 것**: 이 저장소는 외부 모델·클러스터가 없어도 배선을 확인할 수 있도록
> **로컬 대체 모드**를 갖추고 있습니다. 결정 모델(Jev)과 생성 LLM이 설정되지 않으면 어휘
> 겹침·규칙 기반으로 판정하고, 검색은 내장 코퍼스를 메모리에서 색인하며, 생성은
> 검색 문장을 추출해 인용을 붙입니다. 정확도가 아니라 **구조와 배선**을 재현하는
> 모드입니다. 실제 품질 수치는 모델과 인덱스를 붙였을 때의 값과 다릅니다.

---

## 1. 빠른 시작 (모델·클러스터 없이, 2분)

빠른 시작은 설치 범위에 따라 두 묶음입니다(부록 A.7과 같은 순서).

**1-A. 최소 설치로 되는 로컬 실습** — 데모·RRSI 진화·로컬 검증·검색 재현율(부록 A.2~A.5)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install pyyaml pytest                # 최소 설치(코어만)

python3 -m system1.demo                  # 1장: 게이트 4종
python3 rrsi.py --domain rag smoke       # 컴파일·구성·소수 과제 생존
python3 rrsi.py --domain rag baseline    # H_0 평가 → runs/rag/frontier.json
python3 rrsi.py --domain rag --T 1 run   # 진화 1라운드
python3 rrsi.py --domain rag status
python3 tests/local_verify.py            # 다섯 시나리오(클러스터 없이)
python3 -m eval.recall_by_type           # 3장: 질문 유형별 재현율
```

**1-B. 전체 설치 후 전체 시험·회귀 평가** — 격리·인젝션·장애·평가 무결성 시험 313건(부록 A.6), 5절 회귀 평가

```bash
pip install -r requirements.txt pytest==8.3.4   # 전체 설치 + 시험 도구(requirements.txt에는 pytest가 없음)
python3 -m pytest tests -q -rs                  # 313건 수집, 건너뛴 이유도 출력
make eval                                       # 회귀 부분집합 평가(5절)
```

최소 설치에서 `pytest tests`를 실행하면 패키지가 없는 시험은 건너뛰고, `httpx`가 없어
참조 오케스트레이터를 불러오지 못하는 시험 1건이 실패합니다. 전체 시험은 1-B로 확인해 주세요(6절 표).

설치 범위 고르기:

| 하려는 일 | 설치 |
|---|---|
| 1-A 로컬 실습(하네스·RRSI·로컬 검증) | `pip install pyyaml pytest` |
| 1-B 전체 시험·회귀 평가 | `pip install -r requirements.txt pytest==8.3.4` |
| 개발용 토큰 발급(`tools/dev_token.py`, `make token`)만 | `pip install "PyJWT[crypto]==2.10.1"` |
| 서비스·교차 인코더까지 | `pip install -r requirements.txt` (torch 포함 수 GB) |

리눅스에서 `requirements.txt`를 그대로 설치하면 `sentence-transformers`가 CUDA용 torch(수 GB)를
끌어옵니다. GPU가 없으면 CPU용 torch를 먼저 설치한 뒤 나머지를 설치해 주세요(이미 설치된 torch를 그대로 씁니다).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

로컬 모드의 동작:

| 구성 요소 | 로컬 대체 | 실제 |
|---|---|---|
| 결정 모델 | `harness/llm.py`의 어휘·패턴 판정 | `DECISION_BASE_URL` → `/v1/systemone` |
| 생성 | 검색 문장 추출 + 인용 | `OPENAI_BASE_URL`/`OPENAI_API_KEY` |
| 검색 | 내장 코퍼스 BM25 + 해싱 벡터 + RRF | Qdrant(`retrieval.backend: http`) |
| 리랭킹 | 어휘 점수 | `sentence-transformers` 교차 인코더 |

환경 변수로 실제 모델을 붙입니다.

```bash
export DECISION_BASE_URL=http://localhost:3000
export OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-...
python3 rrsi.py --domain rag baseline
```

---

## 2. 저장소 구조 (12.1절)

```text
agentic-rag-lab/
├── rrsi/                     # RRSI 코어 (교육용 독립 구현, 실습 중 고정 영역)
├── rrsi.py                   # CLI: smoke / baseline / run / status / heldout / calibrate
├── system1/                  # 1·4장: 결정 클라이언트·게이트 (python3 -m system1.demo)
│                             # client / types / gates / policy / degrade / local
├── decision_head/            # 5장: 자체 결정 헤드 (학습·보정·평가·서빙)
├── domains/rag/
│   ├── adapter.py            # 도메인 구현 (문항·실행·채점·가드)
│   ├── briefs.py             # 브리프 + 편집 카탈로그
│   ├── judge.py              # 정답 판정·인용 지지 검사
│   ├── outcomes.py           # 기권 사유 분류표(정상 기권·보안 차단·운영 장애·예산 초과)
│   ├── rrsi.json             # 하이퍼파라미터
│   ├── SKILL.md              # 제안자의 헌법
│   ├── PATTERNS.md           # 좋은 편집의 패턴
│   ├── data/                 # 골든셋 42문항·분할·코퍼스 7문서 (읽기 전용)
│   └── harness/              # 유일한 편집면
│       ├── pipeline.py       # 제어 흐름
│       ├── gates.py          # 게이트 문구·임계값
│       ├── retrieve.py       # 하이브리드 검색 도구
│       ├── context.py        # 컨텍스트 조립·인용 표식
│       ├── llm.py            # 결정 모델·생성 LLM 접점 (+로컬 대체)
│       ├── prompts/          # system / rewrite / cite
│       ├── config.yaml       # 진화가 건드리는 수치의 단일 출처
│       └── run.py            # 실행 진입점 (출력 스키마 고정)
├── services/                 # gateway / orchestrator / retriever / reranker /
│                             # generator / guard / decision / ingest / common
│                             # orchestrator/: main + 6장 부품(orchestrate·loop·rewrite·
│                             #   escalate·deadline·injection·policy_store)
├── obs/                      # 10장: 시맨틱 규약·트레이싱·메트릭
├── eval/                     # run_subset · check_thresholds · audit · ragas ·
│                             # recall_by_type · chunking · rewrite · index ·
│                             # bootstrap_delta · gates/(4장 임계값·채점) · ci/
├── deploy/
│   ├── base/                 # 공통 매니페스트 (Kustomize)
│   ├── overlays/kind/        # 로컬 오버레이 (결정 프록시 CPU 1개, :dev 이미지)
│   ├── overlays/gke/         # GKE 오버레이 (GPU·GCE Ingress·Workload Identity)
│   ├── overlays/kind-qdrant/ # kind + Qdrant 백엔드 묶음 (gke-qdrant도 같은 구성)
│   ├── components/qdrant-backend/  # Qdrant 백엔드 전용: 모델 캐시 볼륨·startupProbe·모델 내려받기 egress
│   ├── kind/                 # kind 클러스터 설정, Secret 예시
│   ├── decision/             # 4장: GKE 위 OpenJev(H100) 결정 서비스 (kubectl apply -k), shim/Dockerfile
│   └── observability/        # OTel collector·알림 규칙·KEDA·야간 평가
├── load/                     # k6 시나리오: rag.js · smoke.js · decision-down.js
├── tests/                    # tenant / injection / fault / meta + verify.sh · local_verify.py
├── tools/                    # dev_token · build_goldenset · derive_beta
├── docs/                     # runbooks / drills
├── .github/workflows/        # ci.yml · rag-regression.yml
└── Makefile
```

`services/orchestrator`는 `domains/rag/harness`를 import합니다. 평가에서는 subprocess로,
프로덕션에서는 HTTP 핸들러 안에서 같은 코드를 부릅니다. 그래야 "평가에서 이긴 하네스"가
프로덕션에서 다른 코드로 동작하는 일이 없습니다. 서비스는 가능한 한 얇게 둡니다.

---

## 3. kind에서 전부 띄우기

전제: Docker, kind, kubectl, k6, jq.

```bash
make up          # 이미지 빌드 → 클러스터 → 이미지 적재 → Ingress → 개발용 Secret → 배포 → 대기 (10~15분)
make token       # /tmp/rag-token(demo→acme), /tmp/rag-token-b(other→globex)
make index       # 코퍼스 → 인덱스 스냅샷(JSON). QDRANT_URL이 있으면 Qdrant에도 적재
make smoke
make verify      # 다섯 시나리오
make down
```

설정을 바꾼 뒤에는 `make config`, 검색 백엔드를 Qdrant로 바꿀 때는 `make use-qdrant`를 씁니다
(아래 "설정 바꾸기와 Qdrant 전환").

클러스터 없이 같은 다섯 시나리오는 `make verify-local`, 단위 테스트는 `make test`.

- Ingress 호스트는 `rag.localtest.me`입니다. `*.localtest.me`는 127.0.0.1로 해석됩니다.
- `make up`이 30분을 넘으면 원인은 대개 이미지 풀(`kind load`), 프로브, 리소스
  requests 과다입니다.
- `make up`은 개발용 Secret(`rag-secrets`)을 자동으로 만듭니다. 값을 직접 정하려면
  `deploy/kind/01-secret.example.yaml`을 참고해 `deploy/kind/01-secret.yaml`을 만들어 주세요
  (그 파일이 있으면 그것을 적용합니다). 실제 Secret은 커밋하지 않습니다.
- `deploy/base/networkpolicy.yaml`은 기본 거부 후 필요한 경로만 엽니다. kind의 기본 CNI
  (kindnet)도 NetworkPolicy를 적용합니다(kind v0.33.0에서 gateway→retriever, generator→decision,
  파드→인터넷 차단을 실측). 아주 오래된 kind는 정책 엔진이 없어 규칙이 조용히 무시되므로
  `kubectl -n kube-system logs ds/kindnet | grep -i "policy engine"`으로 먼저 확인해 보세요.
  새 서비스를 더하면 egress 규칙도 함께 더해 주세요. 외부 생성 API를 붙일 때도 해당 파드에
  egress를 열어야 합니다(기본은 클러스터 DNS 외 외부 통신 차단).
- 윈도우에서는 WSL2의 리눅스 셸에서 실행해 주세요. Git Bash의 curl은 한글 인자를 UTF-8이 아닌
  코드 페이지로 보내 `make smoke`·`tests/verify.sh`가 400을 받습니다.
- 이미지는 `services/requirements.txt`(CUDA용 torch를 끌고 오는 sentence-transformers 제외)로
  만듭니다. 서비스 이미지는 약 0.75GB, CPU용 torch를 넣은 reranker만 약 2.2GB입니다.
- kind에는 metrics-server가 없어 `kubectl -n rag get hpa`의 CPU 값이 `<unknown>`으로 보입니다.
  HPA를 실제로 움직여 보려면 metrics-server를 따로 설치해 주세요.
- Qdrant 적재: `kubectl -n rag port-forward svc/qdrant 6333:6333`을 띄운 뒤
  `QDRANT_URL=http://localhost:6333 make index`를 실행하면 컬렉션(스냅샷 이름)에 업서트하고
  별칭 `docs_current`를 옮깁니다. point ID는 chunk_id의 uuid5이고 chunk_id는 payload에 남습니다.
  그다음 검색 백엔드를 Qdrant로 바꿉니다. 순서는 아래 "설정 바꾸기와 Qdrant 전환"을 따라 주세요.
- `kubectl -n rag scale deploy/decision --replicas=0` 상태에서는 결정 모델 호출이 거절되지
  않고 타임아웃(1.5초)까지 기다리므로 요청 하나가 데드라인(약 6초)만큼 걸린 뒤
  `status=abstained`, `reason=g3_unavailable`로 끝납니다. `load/decision-down.js`의 응답 시간도 그 값입니다.
  `tests/verify.sh`의 마지막 시나리오는 decision 파드가 0개가 된 것을 확인하지 못하면 실패하고,
  응답이 200이면 `abstained=true`·`reason=g3_unavailable`일 때만 통과합니다(사유가 없으면 실패).
- 도메인 지표(`rag_requests_total`·`rag_request_duration_seconds`·`rag_tokens_total`·
  `rag_gate_verdicts_total`)는 orchestrator가 응답 직전에 `obs.metrics.record()`로 기록해 OTLP로
  수집기에 보냅니다. kind의 수집기(`deploy/base/otel.yaml`)는 traces·metrics 두 파이프라인 모두
  `debug`(verbosity: basic)로만 내보내므로 `kubectl -n rag logs deploy/otel-collector`에는 받은 건수만
  찍힙니다. 지표 이름·값까지 보려면 verbosity를 `detailed`로 바꾸고, Prometheus로 보내려면
  `deploy/observability/otel-collector-config.yaml`의 `prometheus` 내보내기(8889)를 씁니다.

### 설정 바꾸기와 Qdrant 전환

서비스는 설정을 환경 변수로 읽고, 환경 변수는 파드가 뜰 때 한 번만 정해집니다. 그래서 ConfigMap
값만 바꿔서는 실행 중인 파드에 반영되지 않습니다. 이 저장소는 `rag-config`를 kustomize
`configMapGenerator`로 만들어(원본 `deploy/base/rag-config.env`) 이름 끝에 내용 해시를 붙입니다
(`rag-config-<해시>`). 값을 고치고 `kubectl apply -k`를 다시 실행하면 이름이 바뀌고, 그 이름을
참조하는 Deployment의 파드 템플릿도 바뀌어 롤링 재시작이 일어납니다.

```bash
# deploy/base/rag-config.env 를 고친 뒤
make config                          # 적용 → 재시작 완료 대기 → 파드가 실제로 읽은 값 출력
make config-show KEY=MAX_REWRITES    # 특정 키를 파드에서 다시 확인
make config-prune                    # (선택) 더는 쓰지 않는 예전 rag-config-<해시> 정리
```

검색 백엔드를 Qdrant로 바꿀 때는 순서가 중요합니다. 색인이 비어 있는데 백엔드부터 바꾸면
검색 결과가 0건이 되어 모든 질문이 기권으로 끝납니다.

| 순서 | 하는 일 | 명령 |
|---|---|---|
| 1 | Qdrant에 색인 | `kubectl -n rag port-forward svc/qdrant 6333:6333 &` → `QDRANT_URL=http://localhost:6333 make index` |
| 2 | 색인 성공 확인(별칭 `docs_current`의 점 개수 > 0) | `curl -s -X POST -H 'content-type: application/json' -d '{"exact":true}' http://localhost:6333/collections/docs_current/points/count \| jq .result.count` |
| 3 | `RETRIEVER_BACKEND=qdrant` 적용(Qdrant용 오버레이) | `deploy/base/rag-config.env` 수정 → `kubectl apply -k deploy/overlays/kind-qdrant` |
| 4 | retriever 재시작 완료 대기(모델 내려받기 포함) | `kubectl -n rag rollout status deploy/retriever --timeout=600s` |
| 5 | 실제 backend 확인 | `make backend-check EXPECT=qdrant` |

`make use-qdrant`는 2~5단계를 한 번에 실행하고, 2단계에서 점이 0개이면 설정을 바꾸지 않고 멈춥니다.
5단계는 retriever 파드의 환경 변수와 `/search` 응답의 `backend` 필드를 함께 봅니다(`/healthz`에는
이 필드가 없습니다). `backend` 필드는 프로세스가 읽은 설정값이므로, 결과가 실제로 Qdrant에서
왔는지는 `make smoke`의 인용이 색인한 문서를 가리키는지로 한 번 더 확인해 주세요.
되돌릴 때는 `rag-config.env`의 값을 `local`로 바꾸고 `make config`를 실행합니다.

3단계에서 적용하는 `deploy/overlays/kind-qdrant`(GKE는 `gke-qdrant`)는 기존 오버레이에
`deploy/components/qdrant-backend`를 더한 것입니다. Qdrant 백엔드의 retriever는 임베딩 모델
(기본 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384차원, 약 0.22GB)을 읽어야
검색할 수 있고, 모델이 이미지에 들어 있지 않아 파드가 뜰 때 내려받습니다. 그래서 이 묶음이 세 가지를 더합니다.

| 더하는 것 | 이유 |
|---|---|
| `model-cache` 볼륨(emptyDir, `/models`)과 `FASTEMBED_CACHE_PATH=/models/fastembed`, `HF_HOME=/models/hf` | 컨테이너 루트 파일 시스템이 읽기 전용이라 모델을 둘 쓰기 가능한 자리가 필요합니다 |
| `startupProbe`(`/ready`, 10초 간격 30회 = 최대 5분) | `/ready`는 모델을 다 읽은 뒤에야 통과합니다. 그동안 컨테이너를 다시 시작하지 않고 기다립니다 |
| NetworkPolicy `retriever-model-download-egress`(외부 TCP 443, 사설 대역 제외) | 기본 거부 아래에서는 모델 내려받기가 막힙니다 |

- 기본(메모리 검색) 경로의 `deploy/overlays/kind`·`gke`는 이 묶음을 쓰지 않으므로 렌더링 결과가 그대로입니다.
- `make deploy`·`make config`는 `rag-config.env`의 `RETRIEVER_BACKEND`가 `qdrant`이면 `$(OVERLAY)-qdrant`를,
  아니면 `$(OVERLAY)`를 적용합니다. `kubectl apply -k`를 직접 실행할 때는 백엔드에 맞는 오버레이를 골라 주세요.
- emptyDir은 파드마다 따로 생기므로 retriever 파드는 새로 뜰 때마다 모델을 다시 내려받습니다. 내려받기를
  없애려면 모델을 이미지에 넣고 `EMBED_LOCAL_FILES_ONLY=1`을 주거나 emptyDir 대신 PVC를 붙입니다.
- 모델을 바꾸려면 `rag-config.env`에 `EMBED_MODEL`(필요하면 `EMBED_DIM`)을 더하고, 같은 값으로 `make index`를
  다시 실행해 새 컬렉션에 색인합니다. 색인과 검색의 모델·차원이 다르면 검색이 오류로 끝납니다.
- `make index`는 노트북에서 실행하므로 모델도 노트북의 캐시(`FASTEMBED_CACHE_PATH`, 없으면 임시 폴더)에 내려받습니다.
- 이 경로는 `kubectl kustomize` 렌더링까지만 확인했습니다. 클러스터에서의 내려받기 시간과 메모리 사용량은
  환경에 따라 다르므로, `startupProbe` 횟수와 retriever의 메모리 limit(1Gi)은 실제 값을 보고 조정해 주세요.

### GKE로 옮기기

```bash
# 클러스터와 노드풀 (2장 2.4절, 12.7절) — 일반 풀 1개 + GPU 풀 1개
# 지역(regional) 클러스터에서 --num-nodes·--min-nodes·--max-nodes는 "존당" 값입니다.
# asia-northeast3은 존이 3개라 --num-nodes=2면 6대, --max-nodes=8이면 최대 24대가 됩니다.
# --total-min-nodes·--total-max-nodes는 존당이 아니라 "그 풀 전체" 기준입니다.
#
# 1) 클러스터 + 일반 풀: 클러스터를 만들 때 생기는 기본 풀(default-pool)을 일반 CPU 풀로 그대로 씁니다.
#    general 풀을 따로 더하지 않습니다(더하면 일반 노드가 기본 풀 3대 + general 3대 = 6대로 시작합니다).
#    존당 1대 × 3존 = 처음 3대, 자동 확장은 풀 전체 2~8대.
gcloud container clusters create rag-prod --region asia-northeast3 \
  --release-channel regular --workload-pool=PROJECT.svc.id.goog --enable-ip-alias \
  --enable-dataplane-v2 \
  --machine-type=e2-standard-4 --node-labels=pool=general \
  --num-nodes=1 --enable-autoscaling --total-min-nodes=2 --total-max-nodes=8

# 2) 결정 모델용 GPU 풀(L4 24GB). L4가 있는 a·b 두 존만 씁니다.
#    존당 1대 × 2존 = 처음 2대, 자동 확장은 풀 전체 0~3대.
gcloud container node-pools create decision-gpu \
  --cluster=rag-prod --region=asia-northeast3 \
  --node-locations=asia-northeast3-a,asia-northeast3-b \
  --machine-type=g2-standard-16 --num-nodes=1 \
  --accelerator=type=nvidia-l4,count=1,gpu-driver-version=latest \
  --node-taints=nvidia.com/gpu=present:NoSchedule \
  --enable-autoscaling --total-min-nodes=0 --total-max-nodes=3

kubectl get nodes -L cloud.google.com/gke-nodepool,topology.kubernetes.io/zone   # 풀·존별 노드 수 확인
kubectl apply -k deploy/overlays/gke     # PROJECT, 도메인, 인증서 이름을 먼저 바꿉니다
```

노드 수 계산(지역 클러스터, `--num-nodes`는 존당 · `--total-*`는 풀 전체 기준)

| 풀 | 쓰는 존 | `--num-nodes`(존당) | 처음 노드 수 | 최소(풀 전체) | 최대(풀 전체) |
|---|---|---|---|---|---|
| `default-pool`(일반 CPU, e2-standard-4) | a·b·c (3개) | 1 | 1 × 3 = **3** | 2 | 8 |
| `decision-gpu`(L4, g2-standard-16) | a·b (2개) | 1 | 1 × 2 = **2** | 0 | 3 |
| 합계(2장 구성) | | | **5** | **2** | **11** |
| `decision-h100`(4장에서 추가, a3-highgpu-1g) | c (1개) | 1 | 1 × 1 = 1 | 0 | 2 |
| 합계(4장까지) | | | 6 | 2 | 13 |

- 일반 CPU 노드는 처음 3대, 범위 2~8대입니다. 기본 풀을 두고 `general` 풀을 또 만들면 처음 6대,
  범위 4~14대가 되므로 풀을 하나로 합쳤습니다.
- 기본 풀의 이름은 `default-pool`로 고정됩니다(클러스터 생성 명령으로는 이름을 바꿀 수 없습니다).
  책 그림의 "general 풀"이 이 풀이며, `pool=general` 라벨로 구분할 수 있습니다.
- GPU 풀은 결정 파드가 없으면 자동 확장기가 0대까지 줄입니다. `gpu-driver-version=latest`는
  드라이버를 그 시점의 최신으로 설치하므로, 재현하려면 설치된 버전을 기록해 두세요
  (`kubectl get nodes -l cloud.google.com/gke-accelerator -o yaml | grep -i driver`).

- `--enable-dataplane-v2`: NetworkPolicy를 집행하는 데이터 플레인입니다. 이것(또는
  `--enable-network-policy`)이 없으면 `deploy/base/networkpolicy.yaml`이 조용히 무시됩니다.
  기존 클러스터는 `gcloud container clusters describe rag-prod --region asia-northeast3
  --format="value(networkConfig.datapathProvider)"`가 `ADVANCED_DATAPATH`인지 확인해 주세요.
- GKE 오버레이의 `networkpolicy-gke.yaml`은 base의 ingress-nginx 허용을 지우고 다음을 엽니다.
  GCE 부하 분산기·상태 검사 대역(35.191.0.0/16, 130.211.0.0/22) → gateway 8080,
  Workload Identity 메타데이터 서버(169.254.169.252의 988·987, Dataplane V2용 169.254.169.254의
  80·8080 — [GKE 네트워크 정책 문서](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/network-policy) 기준)와
  Google API(443) → retriever·orchestrator, NodeLocal DNSCache(169.254.20.10:53).
- base의 `allow-metrics-scrape`는 `monitoring`(직접 설치한 Prometheus)·`gmp-system`(관리형
  Prometheus) 네임스페이스에서 8080(`/metrics`) 수집을 엽니다. 막히면 KEDA가 decision 큐 지표를
  받지 못해 자동 확장이 멈춥니다.
- 4장 결정 서비스(`deploy/decision`)의 가중치 PVC는 `standard-rwx`(Filestore Basic HDD, GKE에서
  100GiB부터) 120Gi입니다. `premium-rwx`(Basic SSD)는 최소 2.5TiB라 120Gi 요청과 맞지 않습니다.

이미지 주소를 빼면 달라지는 것은 노드풀(GPU)·아이덴티티·Ingress(GCE + ManagedCertificate)뿐입니다.
base는 그대로 씁니다. `external-secret.yaml`은 External Secrets Operator를 설치하고
`SecretStore`(`gcp-sm`)를 만든 뒤 `deploy/overlays/gke/kustomization.yaml`의 주석을 풀어 적용합니다.
`SecretStore`가 Secret Manager에 인증하는 방법(Workload Identity용 서비스 계정 주석, IAM 바인딩 명령,
적용 뒤 확인 명령)은 `deploy/overlays/gke/secret-store.example.yaml`에 예제로 넣었습니다.
4장 결정 서비스(`deploy/decision`)는 `rag-secrets`에 `SHIM_TOKEN` 키가 있어야 뜨는데,
`external-secret.yaml`은 이 키를 가져오지 않습니다. 4장을 적용할 때는 Secret Manager에 비밀을 하나 더
만들고 `external-secret.yaml`의 `data`에 `SHIM_TOKEN` 항목을 더해 주세요.

---

## 4. RRSI 루프

```bash
python3 rrsi.py --domain rag smoke
python3 rrsi.py --domain rag baseline --job base1
python3 rrsi.py --domain rag baseline --job base2
python3 rrsi.py --domain rag baseline --job base3
python3 rrsi.py --domain rag calibrate --jobs base1,base2,base3
python3 rrsi.py --domain rag --T 1 run
python3 rrsi.py --domain rag status
RAG_HELDOUT_SPLIT=ood_docs python3 rrsi.py --domain rag heldout --label ood_docs
RAG_HELDOUT_SPLIT=ood_intent python3 rrsi.py --domain rag heldout --label ood_intent
```

같은 순서를 `make evolve-baseline`, `make evolve-run T=1`, `make ood`로도 돌릴 수 있습니다.

산출물은 `runs/rag/`에 쌓입니다(커밋하지 않습니다).

```text
runs/rag/
├── frontier.json      # 현재 최고 하네스(커밋·S·C·extra)와 궤적
├── history.jsonl      # 편집별 판정 기록
├── attribution.jsonl  # 승격별 예측-적중 기록
├── calibration.json   # δ 측정 결과
├── jobs/<job>/s<N>/   # 문항별 산출물(답변·인용·판정·토큰)
└── wt/                # 후보 작업장 (git이면 worktree, 아니면 복사본)
```

선택 규칙(7.3절): 하드 가드 → 노이즈 하한 δ → 비용 규칙 → argmax. δ는 같은
하네스를 3회 이상 평가한 점수의 표본 표준편차 σ에 √2를 곱한 값(같은 하네스를 두 번
평가한 점수 차이의 표준편차)에 z(기본 2)를 곱해 얻습니다. 로컬 대체 모드는 결정적이라
δ가 0으로 나올 수 있으므로, 그때는 `rrsi.json`의 `delta_prior`(0.011)를 쓰고
그 사실을 로그에 남깁니다.

### 편집 제안자

- `proposer: catalog` (기본): `briefs.py`의 편집 목록에서 고릅니다. 모델 없이 루프 전체를
  돌리기 위한 방식입니다.
- `proposer: llm`: `OPENAI_BASE_URL`/`OPENAI_API_KEY`가 있으면 LLM이 하네스 파일 한 줄을
  바꾸는 JSON을 냅니다. 경로가 편집면 밖이거나 앵커가 없으면 카탈로그로 되돌아갑니다.

어느 모드든 결과는 (파일, 기존 줄, 새 줄)이고, 적용 전에 critic이 누출·편집면 위반·
죽은 분기를 검사합니다.

---

## 5. 평가

```bash
python3 tools/build_goldenset.py        # 코퍼스 → 골든셋·분할 재생성
python3 eval/audit_goldenset.py         # 분할 중복·정답 개체 미지지 검사
python3 eval/recall_by_type.py --split evolve
python3 eval/chunking/run.py
python3 eval/rewrite/rescue.py
python3 eval/run_subset.py --subset eval/ci/regression_subset.json --out /tmp/report.json
python3 eval/check_thresholds.py /tmp/report.json --item-regressions   # 기준선 비교(9.7절)
python3 -m eval.gates.run_gates           # 4장: G2 임계값 스윕 → eval/reports/gates.json
python3 eval/bootstrap_delta.py --jobs base1,base2,base3   # 9.6절: 위 4절 baseline 산출물(runs/rag/jobs/)로 δ 추정
```

- 골든셋 샘플은 **42문항**(evolve 26 / heldout 8 / ood_docs 4 / ood_intent 4)입니다.
  책의 300문항은 같은 JSONL 형식으로 확장합니다. 형식은 `domains/rag/data/goldenset_v1.jsonl`.
- 보상(8.4절): 정답 가능 문항 `r = 0.65·correct + 0.20·cite_precision + 0.15·answered`,
  기권 문항 `r = 1.0`(정상 기권·보안 차단) / `0.0`. 실행 오류(`error`)와 운영 장애·예산 초과·알 수 없는
  사유의 기권(`domains/rag/outcomes.py` 분류표)은 기권이어도 0점이고 `error_rate`(실패율)에 셉니다.
  진화 채점과 회귀 평가가 같은 함수(`domains/rag/judge.py`의 `score_trial`)를 쓰고, 정답 판정은 답변당
  한 번입니다. 심판 장애는 오답과 구별해 `judge_errors`로 남깁니다(평가기 `rag-eval/5`).
- 가드(8.7절): 인용 정밀도 2포인트 하락, 답변 가능 문항 기권율 10%p 증가, 실패율 0.01 넘게 증가,
  기권 문항 기권율 0.85 미만, p95 25% 초과(50ms 미만은 측정 잡음으로 제외). 심판 장애가 있는 후보도 탈락.
- 회귀 검사(9.7절): `run_subset.py`는 빈 목록·중복 ID·없는 ID를 거부하고(종료 코드 2), 보고서에
  `ids`·`n_unique`·`subset_sha256`·`error_rate`·`judge`·`items`를 남깁니다. `check_thresholds.py`는 문항별
  결과(`items`)가 없거나 `ids`·집계 지표와 맞지 않으면 거절하고, 골든셋·평가기·심판·문항 목록
  해시가 기준선과 다르면 "비교할 수 없음"으로 실패하고, `--item-regressions`를 주면 기준선에서 통과한 문항이
  하나라도 실패할 때 실패합니다. 문항 구성을 일부러 바꿨으면 `make eval-baseline`(`--update-baseline`)으로
  기준선을 새로 등록합니다.
- RAGAS 4지표는 `eval/ragas_metrics.py`에만 의존합니다. `requirements-eval.txt`가 필요합니다.
  심판 LLM(`OPENAI_API_KEY` 또는 `RAGAS_JUDGE_BASE_URL`)이 없으면 계산하지 않고 설정 방법을
  출력한 뒤 종료 코드 2로 끝납니다.
- PR마다 이 회귀 평가를 돌리는 CI 구성(`rag-regression.yml`)과 필수 상태 검사 설정은 6절 "CI"에 있습니다.

샘플 코퍼스가 7문서(acme 6 + globex 1)뿐이라 recall·정답률이 실제보다 높게 나옵니다. 이 숫자는 품질의
근거가 아니라 **파이프라인과 루프가 도는지**의 증거로 활용해 주세요.

---

## 6. 테스트

```bash
python3 -m pytest tests -q     # 수집 313건(전체 설치 기준): 격리 13 + 등급 정책 3 + 인젝션 5 + 장애 3 + 요청 경로 87 + 평가 무결성 177 + 제안자·훈련 9 + 청커 1 + Qdrant·임베딩 13 + OTel 2
python3 tests/local_verify.py  # 정상·기권·인젝션·테넌트·결정모델다운
bash tests/verify.sh           # kind에서 같은 다섯 가지
```

메타 테스트(`tests/meta/leaky_retriever.py`)는 일부러 새는 구현 다섯 가지를 넣고,
격리 테스트가 그것을 잡는지 확인합니다. 테스트가 통과하는 것과 테스트가 의미 있는
것은 다릅니다(3.6절).

수집되는 시험은 313건이고, 통과·건너뜀 수는 설치 환경에 따라 다릅니다(부록 A.6).

| 설치 | 결과 |
|---|---|
| `pip install -r requirements.txt pytest==8.3.4`(전체) | 313건 모두 실행 — 임베딩 모델이 캐시에 없으면 `312 passed, 1 skipped` |
| `pip install pyyaml pytest`(최소) | 패키지가 없는 시험은 건너뛰고, `httpx`가 없어 참조 오케스트레이터를 불러오지 못하는 시험 1건이 실패 |
| 최소 + `httpx==0.28.1` | 실패 없이 일부 건너뜀 |
| CI(`pyyaml pytest fastapi httpx pyjwt`) | 실패 없이 qdrant·fastembed·opentelemetry 시험을 건너뜀 |

313건을 모두 보려면 `pip install -r requirements.txt pytest==8.3.4`를 먼저 실행해 주세요(`requirements.txt`에는 pytest가 없습니다). 건너뛴 이유는 `pytest -rs`로 확인합니다.
실제 임베딩 모델로 33개 청크를 색인하고 검색하는 시험(`tests/qdrant/test_embedding_model.py`)은
모델 파일이 캐시에 있을 때만 실행합니다. `python -m eval.index --qdrant-url :memory:`를 한 번
실행하면 기본 모델(약 0.22GB)이 캐시에 들어가고, `RUN_EMBED_DOWNLOAD=1`을 주면 시험이 직접 내려받습니다.
Qdrant 메모리 모드 적재·검색 시험(`tests/qdrant`, qdrant-client 필요)과 OTel 초기화·추적 전파
시험(`tests/obs`, opentelemetry-sdk·fastapi 필요)도 패키지가 없으면 건너뜁니다.

### CI (`.github/workflows/`)

이 저장소를 GitHub에 올리면(복제본·포크 포함) 두 워크플로가 동작합니다.

| 파일 | 하는 일 |
|---|---|
| `ci.yml` | `pyyaml pytest fastapi httpx pyjwt` 설치 → `pytest tests -q` → 로컬 검증·골든셋 감사 |
| `rag-regression.yml` | PR마다 회귀 부분집합 평가(5절) → 허용치를 벗어나면 CI 실패 |

CI 실패만으로는 병합이 막히지 않습니다. 기본 브랜치의 보호 규칙 또는 Ruleset에서 이 평가 작업을
필수 상태 검사로 설정하면, 평가에 실패한 PR의 병합을 차단할 수 있습니다. 워크플로가 한 번 실행된 뒤
Settings → Rules → Rulesets(또는 Settings → Branches)에서 "Require status checks to pass"를 켜고
job 이름 `eval`(화면 표시 `rag-regression / eval`)을 필수 검사로 추가합니다.

---

## 7. 내 GitHub 저장소로 올리기(ZIP으로 받은 사본)

복제(`git clone`)했다면 GitHub의 Fork를 쓰시면 됩니다. ZIP으로 받은 폴더를 내 저장소로 올릴 때는
다음 순서입니다.

```bash
cd agentic-rag-lab
git init
git add .
git commit -m "agentic-rag-lab: 실습 사본 초기 커밋"

# GitHub에서 빈 저장소를 만든 뒤 (README·.gitignore 추가하지 않습니다)
git remote add origin https://github.com/<사용자>/agentic-rag-lab.git
git branch -M main
git push -u origin main
```

### 장별 태그 (ch01~ch12)

본문 코드와 1:1 대응을 위해 장별 태그를 붙입니다. 장마다 커밋을 나눴다면 마지막
커밋에 태그를 붙이고, 한 번에 올렸다면 같은 커밋에 태그를 붙여 "어느 장의 상태인지"만
표시합니다.

```bash
for ch in $(seq -w 1 12); do
  git tag -a "ch$ch" -m "chapter $ch" HEAD
done
git push origin --tags
```

태그는 나중에 `git checkout ch07`로 그 장 시점의 코드를 볼 수 있게 합니다. 위 반복문처럼
열두 태그를 모두 같은 커밋(HEAD)에 붙이면, 태그는 장별 중간 상태가 아니라 "그 장의 본문이
설명하는 최종 통합 코드"를 표시합니다. 이 저장소는 이 방식이므로 `ch01`~`ch12`가 모두
같은 코드를 가리킵니다.

### 올리기 전 확인

```bash
git status --short          # runs/, .env, secret 이 섞이지 않았는지
python3 -m pytest tests -q
python3 tests/local_verify.py
```

`.gitignore`가 `runs/`, `eval/reports/*.json`, `deploy/**/secret.yaml`, `.env`를 제외합니다.

---

## 8. 재현성 정보

검증 범위. 저자가 확인한 범위는 로컬 대체 모드(시험·`tests/local_verify.py`·평가·진화)와 kind 배포입니다.
GKE 오버레이, 4장 GPU 결정 서비스(`deploy/decision`), `SecretStore`·External Secrets, 원격 Qdrant 묶음은
`kubectl kustomize` 렌더링까지 확인한 준비 예제이며, 실제 GPU 모델 로딩·GKE·Secret Manager 동기화는
독자 환경에서 검증해 주세요. 확인하지 않은 경로를 검증 완료로 표시하지 않습니다.

| 항목 | 고정 값 | 위치 |
|---|---|---|
| 앱 이미지(kind) | `rag/<서비스>:dev` (로컬 빌드) | `deploy/overlays/kind/kustomization.yaml` |
| 앱 이미지(GKE) | orchestrator·gateway `harness-9f2c1ab`, 나머지 `0.9.0` | `deploy/overlays/gke/kustomization.yaml` |
| Qdrant | 서버 `qdrant/qdrant:v1.20.2-unprivileged`, 클라이언트 `qdrant-client==1.20.0` | `deploy/base/qdrant.yaml`, `requirements.txt` |
| OTel Collector | `otel/opentelemetry-collector-contrib:0.115.1` | `deploy/base/otel.yaml` |
| 결정 서빙 | `vllm/vllm-openai:v0.29.0` | `deploy/decision/10-decision.yaml` |
| 결정 API(shim) | `openjev-shim:0.1.0` — **직접 빌드**(공개 이미지 없음). `make shim-image SHIM_REGISTRY=<레지스트리>` → `docker push` → 매니페스트의 `registry.example.com/rag/openjev-shim:0.1.0`을 같은 주소로 교체 | Dockerfile `deploy/decision/shim/Dockerfile`, 사용처 `deploy/decision/10-decision.yaml` |
| 결정 모델 | `openjev/openjev`, 리비전 `MODEL_REVISION`(기본 `main` — **커밋 해시로 바꿔야 재현됩니다**, 아래 경고) | `deploy/decision/05-weights-download.yaml` |
| Ingress(kind) | `ingress-nginx` `controller-v1.15.1` | `Makefile`의 `up` |
| 가중치 다운로드 도구 | `python:3.12-slim` + `huggingface_hub==0.35.3` | 같은 파일 |
| 파이썬 의존성 | 모두 `==` 고정 | `requirements.txt`, `services/requirements.txt`, `requirements-eval.txt` |
| 서비스 기반 이미지 | `python:3.12-slim`(태그 고정, 다이제스트 미고정) | `services/*/Dockerfile` |

> **경고 — 모델 리비전**: `MODEL_REVISION`의 기본값 `main`은 저장소가 갱신되면 다른 가중치와
> helper를 가리킵니다. 책의 수치를 재현하려면 다운로드 Job과 shim 이미지 양쪽에서 같은 커밋 해시
> (40자)로 바꿔 주세요. 해시는 `curl -s https://huggingface.co/api/models/openjev/openjev | jq -r .sha`로
> 조회합니다(2026-11-25 조회 시 `b4a89f5884f3482a5e894b2e7ff01db8ea033726`). shim Dockerfile은 이 해시와
> 그 커밋의 `helper/shim.py` sha256을 기본값으로 고정해 두었고, 값이 다르면 빌드가 실패합니다.

직접 빌드하는 이미지

| 이미지 | Dockerfile | 빌드 명령 |
|---|---|---|
| `rag/<서비스>:dev` 7종(gateway·orchestrator·retriever·reranker·generator·guard·decision) | `services/<서비스>/Dockerfile` | `make image` (`make up`에 포함) |
| `openjev-shim:0.1.0`(4장) | `deploy/decision/shim/Dockerfile` | `make shim-image SHIM_REGISTRY=<레지스트리>` |
| `rag-eval:1.4.0`(야간 평가, `deploy/observability/nightly-eval.yaml`) | 저장소에 없음(자리표시자 `registry.example.com`) | 평가 코드와 `requirements-eval.txt`로 직접 구성 |

태그를 다이제스트로 고정하기. 태그는 같은 이름으로 다시 올라올 수 있으므로, 완전히 같은 이미지를
쓰려면 `이미지:태그@sha256:...` 형태로 적습니다. 아래 다이제스트는 2026-11-25에 Docker Hub에서 조회한
값이며(태그가 실제로 있는지도 함께 확인), 적용 시점에 `docker buildx imagetools inspect <이미지:태그>`로
다시 확인해 주세요.

| 이미지:태그 | 조회한 다이제스트(2026-11-25) | 비고 |
|---|---|---|
| `qdrant/qdrant:v1.20.2-unprivileged` | `sha256:f60765106f26981dabc593d88fe4d0c8138893442c7c59d62257390891b65277` | 버전 태그 |
| `otel/opentelemetry-collector-contrib:0.115.1` | `sha256:d2da12c4336a79758826700be9e21ecf4a9f7d945b7f8a58ba55ee3fa45427c8` | 버전 태그 |
| `vllm/vllm-openai:v0.29.0` | `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` | 버전 태그 |
| `python:3.12-slim` | `sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b` | **움직이는 태그**(패치 버전마다 바뀜). 고정하려면 다이제스트나 `python:3.12.16-slim` 같은 패치 태그 사용 |

매니페스트에 `:latest`·`:main`이나 태그 없는 이미지는 없습니다. 고정되지 않은 것은 위의 `python:3.12-slim`,
모델 리비전 `main`, GPU 드라이버 `gpu-driver-version=latest`(노드풀 생성 명령) 세 가지입니다.

GKE에서 필요한 권한(IAM)

| 주체 | 역할 | 용도 |
|---|---|---|
| 실습자 계정 | `roles/container.admin`, `roles/artifactregistry.writer`, `roles/iam.serviceAccountAdmin` | 클러스터 생성, 이미지 올리기, Workload Identity 연결 |
| GCP 서비스 계정 `rag-runner` | 실제로 호출하는 API의 최소 역할 | retriever·orchestrator 파드 |
| GCP 서비스 계정 `rag-secrets-reader`(External Secrets를 쓸 때) | 비밀별 `roles/secretmanager.secretAccessor`, 쿠버네티스 SA `rag/external-secrets-sa`에 `roles/iam.workloadIdentityUser` | `SecretStore`(`gcp-sm`) 인증. 명령은 `deploy/overlays/gke/secret-store.example.yaml` |
| 쿠버네티스 SA `rag/rag-runner` | GCP SA에 `roles/iam.workloadIdentityUser` 바인딩 | Workload Identity 연결 |
| GKE 노드 서비스 계정 | `roles/artifactregistry.reader` | 이미지 풀 |

## 9. 보안·운영 메모

- 테넌트는 JWT claim에서만 옵니다. 본문·헤더의 tenant_id를 믿지 않습니다.
- 검색 필터는 `TenantScope` 없이는 호출할 수 없습니다. 필터를 인자로 매번 넘기는 방식은
  한 번 빠뜨리면 조용히 전체를 반환합니다.
- G3가 죽으면 **기권**합니다(fail-closed). G1이 죽으면 사내 검색으로 폴백합니다.
- 프롬프트에 카나리(`ZQ7-4f19-KLM`)를 두고 출력 가드가 누출을 차단합니다.
- 운영 비밀은 External Secrets로 가져옵니다. `deploy/kind/01-secret.example.yaml`은 예시입니다.
- 런북: `docs/runbooks/` (롤백·결정 모델 다운·검색 품질), 훈련: `docs/drills/`.

## 10. 라이선스

이 저장소의 코드는 **Apache License 2.0**으로 제공합니다(`LICENSE`, `NOTICE`). 사용·수정·재배포가
가능하며, 재배포하실 때는 `LICENSE`와 `NOTICE`를 함께 두어 주세요.

- 책의 본문 글·도판·표는 이 라이선스의 대상이 아닙니다(책 12장 판권 안내를 따릅니다).
- 내려받아 쓰는 제3자 구성 요소와 모델은 각자의 라이선스를 따릅니다. OpenJev 가중치는
  CC BY-NC 4.0(비상업)이므로 코드 라이선스와 따로 확인해 주세요.
