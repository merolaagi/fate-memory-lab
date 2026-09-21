#!/bin/bash
cd "$(dirname "$0")"
PORT="${FML_PORT:-47431}"
HOST="${FML_HOST:-127.0.0.1}"
exec .venv/bin/uvicorn app.server:app --host "$HOST" --port "$PORT"
