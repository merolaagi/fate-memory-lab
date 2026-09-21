#!/bin/bash
set -e
cd "$(dirname "$0")"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "Python 3.10+ not found. Install it with: brew install python@3.12"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' || { echo "Python 3.10+ is required. Install it with: brew install python@3.12"; exit 1; }
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
mkdir -p data/runs logs
chmod +x run.sh service.sh update.sh
echo "Installed Fate Memory Lab $(cat VERSION) with $("$PY" --version)"
