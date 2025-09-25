#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/huggingface

exec uvicorn app:app --host 0.0.0.0 --port 7860


