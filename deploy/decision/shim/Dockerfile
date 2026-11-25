# syntax=docker/dockerfile:1.7
# OpenJev 결정 API(shim) 이미지 (4.3절). deploy/decision/10-decision.yaml의 shim 컨테이너가 씁니다.
# 공개 레지스트리에 올라온 이미지가 없으므로 직접 빌드해 자신의 레지스트리에 올립니다.
#
#   make shim-image SHIM_REGISTRY=asia-northeast3-docker.pkg.dev/PROJECT/rag
#   docker push asia-northeast3-docker.pkg.dev/PROJECT/rag/openjev-shim:0.1.0
#   → 10-decision.yaml의 image(registry.example.com/rag/openjev-shim:0.1.0)를 같은 주소로 바꿉니다.
#
# shim 본체(helper/shim.py)는 모델 저장소에서 "고정한 커밋"의 파일을 그대로 가져오고,
# 서빙 가이드(serve/SERVE.md)에 적힌 sha256과 다르면 빌드가 실패합니다.
#   - MODEL_REVISION: 05-weights-download.yaml의 MODEL_REVISION과 같은 커밋 해시를 씁니다.
#     아래 기본값은 2026-11-25에 조회한 main의 해시입니다.
#   - SHIM_SHA256: 그 커밋의 helper/shim.py 해시(같은 날 직접 계산해 SERVE.md 값과 일치 확인).
#     커밋을 바꾸면 이 값도 함께 바꿉니다.
# 의존성: SERVE.md가 측정에 썼다고 밝힌 openai 3.16.2·httpx 0.28.1을 고정합니다.
#   transformers는 SERVE.md가 버전을 밝히지 않았습니다(확정 필요). 기본값은 2026-11-25의 최신판이며,
#   토크나이저를 읽는 데만 씁니다. 실제 서빙에서 /v1/version 응답을 확인한 버전으로 바꿔 주세요.
FROM python:3.12-slim
ARG MODEL_REPO=openjev/openjev
ARG MODEL_REVISION=b4a89f5884f3482a5e894b2e7ff01db8ea033726
ARG SHIM_SHA256=81a22f1b1b8912a465059207ef9f60b7c6c16b4de6372305d867efbe38a1987a
ARG OPENAI_VERSION=3.16.2
ARG HTTPX_VERSION=0.28.1
ARG TRANSFORMERS_VERSION=5.24.1
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp HF_HOME=/tmp/hf HF_HUB_OFFLINE=1
RUN pip install --no-cache-dir \
      "openai==${OPENAI_VERSION}" "httpx==${HTTPX_VERSION}" "transformers==${TRANSFORMERS_VERSION}"
ADD --checksum=sha256:${SHIM_SHA256} --chmod=0444 \
    https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/helper/shim.py /opt/openjev/helper/shim.py
LABEL org.opencontainers.image.title="openjev-shim" \
      openjev.model.repo="${MODEL_REPO}" openjev.model.revision="${MODEL_REVISION}"
USER 1000
EXPOSE 3001
# 매니페스트의 args(--host 127.0.0.1 --port 3001)가 뒤에 붙습니다. 토크나이저는 TOKENIZER
# (/models/openjev, PVC)에서 읽으므로 HF_HUB_OFFLINE=1로 외부 내려받기를 막아 둡니다.
ENTRYPOINT ["python", "/opt/openjev/helper/shim.py"]
