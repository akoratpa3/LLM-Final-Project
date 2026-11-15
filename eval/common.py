"""Shared configuration, prompt construction, generation, and scoring."""
import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.text_preprocess import _split_solution, compose_text  # noqa: E402

CKPT_DIR = ROOT / "ckpt"
GEN_DIR = ROOT / "results" / "generations"
TEST_SET_PATH = ROOT / "test_set_final.jsonl"

PRETRAIN_REPO = "Interplay-LM-Reasoning/extrapolation_rl"
PRETRAIN = "id2-10_0.2easy_0.3medium_0.5hard"
# Run name -> checkpoint subfolder inside PRETRAIN_REPO.
CHECKPOINTS = {
    "ar-100m_base": f"{PRETRAIN}/base",
    "ar-100m_rl-op7-10": f"{PRETRAIN}/rl/op7-10_uniform",
    "ar-100m_rl-op9-12": f"{PRETRAIN}/rl/op9-12_uniform",
    "ar-100m_rl-op11-14": f"{PRETRAIN}/rl/op11-14_uniform",
    "ar-100m_rl-op17-20": f"{PRETRAIN}/rl/op17-20_uniform",
}
BANDS = {
    "ID (2-10)": range(2, 11),
    "edge (11-14)": range(11, 15),
    "hard (15-20)": range(15, 21),
}


def make_prompt(example):
    """Question prefix in the authors' format, ending at the solution tag."""
    return compose_text(example).split("<solution>")[0] + "<solution>"


def load_test_set():
    with open(TEST_SET_PATH) as f:
        return [json.loads(line) for line in f]


def gold_answer(solution):
    return _split_solution(solution)[1] or None


def pred_answer(text):
    match = re.search(r"<answer>\s*(-?[\d\.]+)", text)
    if match:
        return match.group(1).rstrip(".")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def generate_ar(model, tok, test_set, run_name, batch_size=32, max_new_tokens=1536):
    """Greedy-decode every problem and write one JSON line per generation."""
    device = next(model.parameters()).device
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEN_DIR / f"{run_name}.jsonl"
    with open(out_path, "w") as f:
        for i in range(0, len(test_set), batch_size):
            batch = test_set[i : i + batch_size]
            enc = tok([make_prompt(e) for e in batch], return_tensors="pt",
                      padding=True, return_token_type_ids=False).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    use_cache=True, pad_token_id=tok.pad_token_id,
                    stop_strings=["</answer>"], tokenizer=tok,
                )
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=False)
            for e, text in zip(batch, texts):
                f.write(json.dumps({"id": e["id"], "op": e["op"], "model": run_name,
                                    "sample_idx": 0,
                                    "gen_solution_answer": "<solution>" + text}) + "\n")
            print(f"{min(i + batch_size, len(test_set))}/{len(test_set)}", flush=True)
    return out_path


def score(test_set, gen_path):
    """Return accuracy per operation band."""
    with open(gen_path) as f:
        gens = [json.loads(line) for line in f]
    rows = [(e["op"], pred_answer(g["gen_solution_answer"]) == gold_answer(e["solution"]))
            for e, g in zip(test_set, gens)]
    accuracy = {}
    for band, ops in BANDS.items():
        hits = [ok for op, ok in rows if op in ops]
        accuracy[band] = {"correct": sum(hits), "total": len(hits),
                          "accuracy": sum(hits) / len(hits)}
    return accuracy
