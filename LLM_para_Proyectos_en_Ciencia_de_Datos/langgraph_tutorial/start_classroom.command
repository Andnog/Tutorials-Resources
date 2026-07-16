#!/bin/zsh
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  uv sync --extra dev
fi
./.venv/bin/streamlit run classroom_app.py --server.port 8502 &
server_pid=$!
sleep 2
open "http://127.0.0.1:8502"
wait "$server_pid"
