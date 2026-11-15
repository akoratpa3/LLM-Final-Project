"""Login-node step: fetch checkpoints (and the test set if absent). Compute nodes are offline.

    uv run python eval/download.py [run_name ...]
"""
import sys
from itertools import islice

import json
from datasets import load_dataset
from huggingface_hub import snapshot_download

from common import CHECKPOINTS, CKPT_DIR, PRETRAIN_REPO, TEST_SET_PATH

OPS = range(2, 21)
PROBLEMS_PER_OP = 100
STREAM_LIMIT = 500_000


def build_test_set():
    """Stream the composition test split and keep the first PROBLEMS_PER_OP problems per op."""
    stream = load_dataset("Interplay-LM-Reasoning/composition", split="test", streaming=True)
    by_op = {op: [] for op in OPS}
    for ex in islice(stream, STREAM_LIMIT):
        if ex["op"] in by_op and len(by_op[ex["op"]]) < PROBLEMS_PER_OP:
            by_op[ex["op"]].append({k: ex[k] for k in
                                    ["id", "op", "problem", "question", "solution"]})
        if all(len(v) >= PROBLEMS_PER_OP for v in by_op.values()):
            break
    with open(TEST_SET_PATH, "w") as f:
        for op in OPS:
            for ex in by_op[op]:
                f.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    if not TEST_SET_PATH.exists():
        build_test_set()
    for name in sys.argv[1:] or CHECKPOINTS:
        snapshot_download(PRETRAIN_REPO, allow_patterns=[f"{CHECKPOINTS[name]}/*"],
                          local_dir=CKPT_DIR)
        print(f"[download] {name}")
