#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ] && \
   .venv/bin/python -c "import fastapi, sounddevice, faster_whisper" >/dev/null 2>&1; then
  exec .venv/bin/python server.py "$@"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it: brew install uv" >&2
  echo "Or set up manually: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating virtualenv..."
  uv venv --python 3.12 .venv >/dev/null
fi

echo "[setup] installing dependencies..."
uv pip install -r requirements.txt

exec .venv/bin/python server.py "$@"
