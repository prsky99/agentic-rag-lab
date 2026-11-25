FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY services/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# 교차 인코더(RERANK_MODEL을 지정했을 때만 사용)는 CPU용 torch로 충분합니다.
# 기본 PyPI의 torch는 CUDA 라이브러리를 함께 받아 이미지가 수 GB 커집니다.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install --no-cache-dir sentence-transformers==3.3.1
COPY services /app/services
COPY domains/rag /app/domains/rag
ENV PYTHONPATH=/app:/app/domains/rag
EXPOSE 8080
CMD ["uvicorn", "services.reranker.main:app", "--host", "0.0.0.0", "--port", "8080"]
