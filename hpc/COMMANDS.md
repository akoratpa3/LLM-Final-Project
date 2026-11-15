# HPC Commands

The commands below reproduce the baseline results in `baseline_results.md`. All commands are run from the repository root.

## Setup (login node)

Compute nodes have no internet access, so the environment installation and all downloads take place on the login node.

```
bash hpc/setup.sh
```

The script runs `uv sync`, downloads the five checkpoints and the test set (if absent), and verifies that transformers is version 4.55.0.

## Evaluation (GPU node)

The evaluation of all five checkpoints was submitted as one job (job 9188309, one A100 GPU, 8 CPUs, 32 GB of memory, 27 minutes 53 seconds).

```
sbatch hpc/gpu.slurm eval/evaluate.py
```

A single checkpoint can be evaluated with its run name as an argument.

```
sbatch --job-name=rl9-12 hpc/gpu.slurm eval/evaluate.py ar-100m_rl-op9-12
```

The run names are ar-100m_base, ar-100m_rl-op7-10, ar-100m_rl-op9-12, ar-100m_rl-op11-14, and ar-100m_rl-op17-20.

## Monitoring and accounting

```
squeue -j <jobid>
sacct -j <jobid> -o State,Elapsed,MaxRSS -X -n
sacct -j <jobid> -X -n -o NodeList,AllocTRES%60
scancel <jobid>
```

## Outputs

Logs are written to `logs/gpu.slurm_<jobid>.out` and `logs/gpu.slurm_<jobid>.err`. Generations and the accuracy summary are written to `results/generations/`, with one JSONL file per run and `summary_all.json` for the combined scores.
