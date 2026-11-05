---
title: Omni Server
emoji: 🛰️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Omni — private AI agent backend

FastAPI + MongoDB (embedded in the image) + tool loop (`run_command`, files, providers).
Auth is a passcode → JWT. Configure secrets: `APP_PASSCODE`, `JWT_SECRET`, `MONGO_URL`, `DB_NAME`, `APP_NAME`, and optionally `DEEPSEEK_API_KEY` / `DEFAULT_MODEL`.
