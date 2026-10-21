# SKELETON — implemented in Sprint 6 (docs/sprints/SPRINT-06-docker.md).
# Target: slim, non-root Python 3.11 image for services/data-pipeline.
#
# FROM python:3.11-slim
# WORKDIR /app
# COPY services/data-pipeline/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY services/data-pipeline/ .
# RUN useradd -m solia && chown -R solia /app
# USER solia
# CMD ["python", "-m", "solia_pipeline"]
