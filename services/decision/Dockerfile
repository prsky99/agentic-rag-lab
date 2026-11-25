FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY services/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY services /app/services
COPY domains/rag /app/domains/rag
ENV PYTHONPATH=/app:/app/domains/rag
# 파드의 runAsNonRoot를 만족하도록 숫자 UID로 실행합니다
RUN useradd --uid 1000 --no-create-home app
USER 1000
EXPOSE 8080
CMD ["uvicorn", "services.decision.main:app", "--host", "0.0.0.0", "--port", "8080"]
