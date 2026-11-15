#!/usr/bin/env bash
# One GPU for any pipeline script. Run hpc/setup.sh on the login node first.
# Site settings (account, partition) can be overridden through SBATCH_* environment variables.
#
#   sbatch hpc/gpu.slurm eval/evaluate.py
#   sbatch --job-name=rl9-12 hpc/gpu.slurm eval/evaluate.py ar-100m_rl-op9-12
#SBATCH --account=def-vganesh
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1  # compute nodes are offline
source .venv/bin/activate
python "$@"
