# SKELETON — implemented in Sprint 6 (docs/sprints/SPRINT-06-docker.md).
# Target: Python 3.11 + PyTorch image for services/ai-service (FastAPI).
#
# FROM python:3.11-slim
# WORKDIR /app
# COPY services/ai-service/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY services/ai-service/ .
# RUN useradd -m solia && chown -R solia /app
# USER solia
# EXPOSE 8000
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
