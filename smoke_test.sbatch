#!/bin/bash
# Person B: Slurm job for the GRPO smoke test (10 RL steps on op 11-14, 1 A100).
#
# Submit from the repo root:
#   sbatch scripts/person_b/smoke_test.sbatch
# Watch it:
#   squeue -u $USER          (is it running?)
#   tail -f logs/smoke_test-<jobid>.out
#
# Edit the lines marked EDIT for your cluster.
#SBATCH --job-name=b-grpo-smoke
#SBATCH --gres=gpu:1                 # EDIT if your cluster names GPUs, e.g. --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # verl starts Ray + vLLM; 32G may be tight
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_test-%j.out
##SBATCH --partition=EDIT            # EDIT: uncomment and set if your cluster needs a partition
##SBATCH --account=EDIT              # EDIT: uncomment and set if your cluster needs an account

set -euo pipefail
mkdir -p logs

# EDIT: activate the RL environment built from requirements_rl.txt
source .venv-rl/bin/activate

echo "== Node: $(hostname) | Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 1. Environment check (stops here if a version is wrong)
python scripts/person_b/check_env.py

# 2. Data + base checkpoint (skips files that are already downloaded)
python scripts/person_b/prepare_data.py

# 3. The smoke test itself
NUM_GPUS=1 bash scripts/person_b/smoke_test.sh
