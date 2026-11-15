#!/usr/bin/env bash
# Login-node setup: install the environment and download checkpoints. Compute nodes have
# no internet, so every download happens here.
#
#   bash hpc/setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync
mkdir -p logs temp results
[ -f ~/.hf_token ] && export HF_TOKEN="$(cat ~/.hf_token)"
uv run python eval/download.py
uv run python -c "import transformers; assert transformers.__version__ == '4.55.0'"
echo "[setup] done"
