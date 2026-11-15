"""GPU step: greedy-decode the test set with each checkpoint and report accuracy per band.

    python eval/evaluate.py [run_name ...]
"""
import gc
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (CHECKPOINTS, CKPT_DIR, GEN_DIR, generate_ar, load_test_set, score)

if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Test set: {len(test_set)} problems")
    summary = {}
    for name in sys.argv[1:] or CHECKPOINTS:
        path = CKPT_DIR / CHECKPOINTS[name]
        print(f"\n===== {name} =====", flush=True)
        tok = AutoTokenizer.from_pretrained(path)
        # float32: bfloat16 lowered accuracy in early tests.
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).cuda().eval()
        summary[name] = score(test_set, generate_ar(model, tok, test_set, name))
        for band, s in summary[name].items():
            print(f"{band}: {s['correct']}/{s['total']} = {s['accuracy']:.0%}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with open(GEN_DIR / f"summary_{'_'.join(sys.argv[1:]) or 'all'}.json", "w") as f:
        json.dump(summary, f, indent=2)
