#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/huggingface

if [[ "${RUNPOD_SERVERLESS:-}" != "" || "${RUNPOD_POD_ID:-}" != "" ]]; then
  python3 rp_handler.py
else
  exec uvicorn app:app --host 0.0.0.0 --port 7860 --access-log --proxy-headers --forwarded-allow-ips="*"
fi


