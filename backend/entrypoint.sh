#!/usr/bin/env bash
set -e
mkdir -p /data/db /app/uploads
mongod --dbpath /data/db --bind_ip 127.0.0.1 --port 27017 --fork --logpath /tmp/mongod.log || true
for i in $(seq 1 30); do
  if (echo > /dev/tcp/127.0.0.1/27017) 2>/dev/null; then break; fi
  sleep 1
done
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-7860}" --workers 1
