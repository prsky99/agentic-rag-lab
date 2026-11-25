FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY services/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY services /app/services
COPY domains/rag /app/domains/rag
# 도메인 지표(obs/metrics.py). orchestrator가 응답 직전에 record()를 부릅니다(10.3절).
COPY obs /app/obs
ENV PYTHONPATH=/app:/app/domains/rag
EXPOSE 8080
CMD ["uvicorn", "services.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8080"]
