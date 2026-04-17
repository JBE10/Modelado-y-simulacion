#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/streamlit ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -U pip -q
  .venv/bin/pip install -r requirements.txt -q
fi

cd python
exec ../.venv/bin/streamlit run dashboard.py \
  --server.address=127.0.0.1 \
  --server.port=8501
