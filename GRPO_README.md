# Person B: RL setup scripts

## Quick path on a Slurm cluster

1. Clone the team repo and build a **separate** Python 3.12 environment with the exact versions in `requirements_rl.txt` (instructions at the top of that file). Do not reuse an environment with newer PyTorch or transformers.
2. From the repo root: `sbatch scripts/person_b/smoke_test.sbatch` (edit the `EDIT` lines first: GPU name, partition/account, environment path).
3. Send back `logs/smoke_test-<jobid>.out`.

The job runs the environment check, downloads the data and base checkpoint, then runs 10 GRPO steps on 1 A100.

## Run order (manual, with conda)

1. **Clone the team repo and set up the environment** (once). Check the CUDA version with `nvidia-smi` first; use a scratch/data disk, not a small home directory.
   ```bash
   git clone https://github.com/akoratpa3/LLM-Final-Project.git
   cd LLM-Final-Project
   CUDA_TAG=cu128 bash scripts/person_b/setup_env.sh
   conda activate physics_rl
   ```
   All later commands run from this repo folder.
2. **Check it works.** Every line should say `OK`.
   ```bash
   python scripts/person_b/check_env.py
   ```
3. **Download the RL data and base checkpoint.**
   ```bash
   python scripts/person_b/prepare_data.py
   ```
4. **Preview, then run the smoke test** (10 GRPO steps on op 11-14).
   ```bash
   NUM_GPUS=1 bash scripts/person_b/smoke_test.sh --dry-run
   NUM_GPUS=1 bash scripts/person_b/smoke_test.sh
   ```

## What success looks like

- Training runs all 10 steps without crashing.
- The reward (`critic/score/mean` in the log) starts near the base model's edge accuracy (about 0.25) and trends upward.
- Response length stays stable.

## Requirements learned the hard way

- `transformers==4.55.0`. Version 5.x loads the checkpoints without error but drops accuracy from 94% to 33%.
- Prompt format: `<question> {problem} {question} </question> <solution>` (the repo's `compose_prompt`).

## If something fails

| Error mentions | Likely fix |
|---|---|
| `flash_attn` | `pip install flash-attn --no-build-isolation`, or add `actor_rollout_ref.model.use_remove_padding=False` to `VERL_EXTRA_ARGS` |
| CUDA / driver mismatch | Rerun `setup_env.sh` with the `CUDA_TAG` matching `nvidia-smi` |
| Out of memory | Lower `actor_rollout_ref.rollout.gpu_memory_utilization` (e.g. 0.4) or `ppo_micro_batch_size_per_gpu` to 4 |
| batch size not divisible | Keep `NUM_GPUS` at 1, 2, 3, 4, 6 or 8 |
| `wandb` login | The script already uses console logging only; check nothing else sets `trainer.logger` |
