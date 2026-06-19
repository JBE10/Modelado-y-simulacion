#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Ensure venv exists and has dependencies
if [[ ! -x .venv/bin/streamlit ]]; then
  echo "Creating virtual environment and installing dependencies..."
  python3 -m venv .venv
  .venv/bin/python -m pip install -U pip -q
  .venv/bin/pip install -r requirements.txt -q
fi

# Run the new modular dashboard
echo "Launching Modular Dashboard (rocket_pro_dashboard.py)..."
exec .venv/bin/python -m streamlit run python/rocket_pro_dashboard.py \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --browser.gatherUsageStats=false
