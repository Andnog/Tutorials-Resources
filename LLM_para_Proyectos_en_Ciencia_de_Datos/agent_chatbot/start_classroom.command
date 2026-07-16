#!/bin/zsh
set -e
cd "$(dirname "$0")"
uv run streamlit run app.py --server.port 8503
