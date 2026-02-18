#!/usr/bin/env python
"""
DTDR Emergence Experiment

Measures how model behaviour appears as coefficients accumulate.
Produces:
 - console samples
 - CSV metrics
 - plotted curve
"""

import argparse
import pickle
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# DTDR reconstruction
# ============================================================

def decompress_blockwise(payload: dict, fraction: float = 1.0) -> torch.Tensor:
    blocks = payload["data"]
    metadata = payload["metadata"]
    bits = int(payload["bits"])
    levels = 2 ** bits

    total = len(blocks)
    keep = max(1, int(total * fraction))

    out = []

    for i, (q, (tmin, tmax, shape, *_)) in enumerate(zip(blocks, metadata)):
        if i < keep:
            scaled = q.astype(np.float32) / (levels - 1)
            deq = scaled * (tmax - tmin) + tmin
        else:
            deq = np.zeros(np.prod(shape), dtype=np.float32)

        out.append(deq.reshape(shape))

    return torch.from_numpy(np.vstack(out)).to(torch.float16)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def overwrite_from_pkl(model, pkl, fraction=1.0):
    for name, param in model.named_parameters():
        if name not in pkl:
            continue

        entry = pkl[name]

        if entry["type"] == "compressed_blockwise":
            w = decompress_blockwise(entry["data"], fraction)
        elif entry["type"] == "uncompressed":
            w = torch.from_numpy(entry["data"]).to(torch.float16)
        else:
            continue

        with torch.no_grad():
            param.data = w.to(param.device)


# ============================================================
# metrics
# ============================================================

@torch.inference_mode()
def get_logits(model, tok, prompt, device):
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model(**inputs)
    return out.logits[0, -1].float().cpu()


def cosine(a, b):
    return F.cosine_similarity(a, b, dim=0).item()


def topk_overlap(a, b, k=10):
    ta = torch.topk(a, k).indices
    tb = torch.topk(b, k).indices
    return len(set(ta.tolist()) & set(tb.tolist())) / k


@torch.inference_mode()
def generate(model, tok, prompt, device):
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--cache-dir", default="./hf_cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("Loading reference model...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device, cache_dir=args.cache_dir
    )

    pkl = load_pkl(args.pkl)
    overwrite_from_pkl(model_ref, pkl, 1.0)

    prompt = "Explain why the sky is blue in simple terms."

    ref_logits = get_logits(model_ref, tok, prompt, device)
    print("\nREFERENCE:\n", generate(model_ref, tok, prompt, device))

    fractions = [0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.0]

    rows = []

    for f in fractions:
        print(f"\n=== Fraction {f*100:.1f}% ===")

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=device, cache_dir=args.cache_dir
        )

        overwrite_from_pkl(model, pkl, f)

        logits = get_logits(model, tok, prompt, device)

        cos = cosine(ref_logits, logits)
        overlap = topk_overlap(ref_logits, logits)

        print("Cosine:", cos)
        print("Top10:", overlap)
        print(generate(model, tok, prompt, device)[:300])

        rows.append((f, cos, overlap))

        del model
        torch.cuda.empty_cache()

    # save CSV
    with open("emergence_curve.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fraction","cosine","top10_overlap"])
        writer.writerows(rows)

    # plot
    x=[r[0] for r in rows]
    y=[r[1] for r in rows]

    plt.figure(figsize=(7,5))
    plt.plot(x,y,marker='o')
    plt.xlabel("Fraction of DTDR coefficients used")
    plt.ylabel("Cosine similarity to full model")
    plt.title("Emergence of inference from distributed coefficients")
    plt.grid(True)
    plt.savefig("emergence_curve.png", dpi=150)

    print("\nSaved emergence_curve.csv and emergence_curve.png")


if __name__ == "__main__":
    main()
