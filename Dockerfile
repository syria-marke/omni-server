# Self-hosted Omni backend (FastAPI + MongoDB inside one container)
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg procps && \
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
      gpg --dearmor -o /usr/share/keyrings/mongodb.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/mongodb.gpg] http://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" \
      > /etc/apt/sources.list.d/mongodb.list && \
    apt-get update && apt-get install -y --no-install-recommends mongodb-org-server && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY backend/ /app/
RUN mkdir -p /data/db /app/uploads

ENV MONGO_URL=mongodb://127.0.0.1:27017 DB_NAME=omni_data PORT=7860 HOME=/app
EXPOSE 7860
CMD ["bash", "/app/entrypoint.sh"]
