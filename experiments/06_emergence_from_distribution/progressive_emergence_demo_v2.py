#!/usr/bin/env python

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

def decompress_blockwise(payload, fraction=1.0, mode="zero", anchor_full=None, rng=None):

    blocks = payload["data"]
    metadata = payload["metadata"]
    bits = int(payload["bits"])
    levels = 2 ** bits

    total = len(blocks)
    keep = max(1, int(total * fraction))

    out = []

    if rng is None:
        rng = np.random.default_rng(12345)

    # estimate noise scale for random fill
    sigma = 1.0
    if mode == "random":
        vals = []
        for i in range(min(keep, 6)):
            q = blocks[i]
            tmin, tmax, shape, *_ = metadata[i]
            scaled = q.astype(np.float32) / (levels - 1)
            deq = scaled * (tmax - tmin) + tmin
            vals.append(deq.reshape(-1))
        if vals:
            sigma = float(np.std(np.concatenate(vals))) or 1.0

    # row tracking across stacked blocks
    row_off = 0

    for i, (q, (tmin, tmax, shape, *_)) in enumerate(zip(blocks, metadata)):
        n = int(np.prod(shape))

        # ------------------- normal reconstruction -------------------
        if i < keep:
            scaled = q.astype(np.float32) / (levels - 1)
            deq = scaled * (tmax - tmin) + tmin

        # ------------------- missing blocks --------------------------
        else:

            if mode == "zero":
                deq = np.zeros(n, dtype=np.float32)

            elif mode == "random":
                deq = rng.normal(0.0, sigma, size=n).astype(np.float32)

            elif mode == "guided":
                if anchor_full is None:
                    raise ValueError("guided mode requires anchor_full")

                # determine rows/cols in this block
                if len(shape) >= 2:
                    r, c = shape[0], shape[1]
                else:
                    r, c = 1, n

                anchor_block = anchor_full[row_off:row_off + r, :c].cpu().numpy().astype(np.float32)
                deq = (0.15 * anchor_block).reshape(-1)[:n]

            else:
                raise ValueError(f"Unknown mode: {mode}")

        out.append(deq.reshape(shape))

        # advance row offset
        if len(shape) >= 2:
            row_off += shape[0]
        else:
            row_off += 1

    return torch.from_numpy(np.vstack(out)).to(torch.float16)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def decode_param(entry, fraction, mode, anchor_full=None, rng=None):
    if entry["type"] == "compressed_blockwise":
        return decompress_blockwise(entry["data"], fraction, mode, anchor_full, rng)
    elif entry["type"] == "uncompressed":
        return torch.from_numpy(entry["data"]).to(torch.float16)
    else:
        return None


def build_anchor_cache(model, anchor_pkl):
    cache = {}
    for name, param in model.named_parameters():
        if name not in anchor_pkl:
            continue
        entry = anchor_pkl[name]
        w = decode_param(entry, 1.0, "zero")
        if w is not None:
            cache[name] = w
    return cache


def overwrite_from_pkl(model, pkl, fraction=1.0, mode="zero", anchor_cache=None, seed=12345):

    rng = np.random.default_rng(seed)

    for name, param in model.named_parameters():
        if name not in pkl:
            continue

        entry = pkl[name]

        anchor_full = None
        if mode == "guided" and anchor_cache and name in anchor_cache:
            anchor_full = anchor_cache[name]

        w = decode_param(entry, fraction, mode, anchor_full, rng)
        if w is None:
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
    ap.add_argument("--anchor-pkl", default=None)
    ap.add_argument("--mode", choices=["zero","random","guided"], default="zero")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("Loading reference model...")
    model_ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)

    pkl = load_pkl(args.pkl)

    anchor_cache = None
    if args.mode == "guided":
        if not args.anchor_pkl:
            raise SystemExit("guided mode requires --anchor-pkl")
        print("Loading anchor PKL...")
        anchor_pkl = load_pkl(args.anchor_pkl)
        print("Decoding anchor tensors...")
        anchor_cache = build_anchor_cache(model_ref, anchor_pkl)

    overwrite_from_pkl(model_ref, pkl, 1.0, mode="zero")

    prompt = "Explain why the sky is blue in simple terms."
    ref_logits = get_logits(model_ref, tok, prompt, device)

    print("\nREFERENCE:\n", generate(model_ref, tok, prompt, device))

    fractions = [0.2, 0.4, 0.6]

    for f in fractions:
        print(f"\n=== Fraction {f*100:.1f}% | mode={args.mode} ===")

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)

        overwrite_from_pkl(model, pkl, f, mode=args.mode, anchor_cache=anchor_cache)

        logits = get_logits(model, tok, prompt, device)

        cos = cosine(ref_logits, logits)
        overlap = topk_overlap(ref_logits, logits)

        print("Cosine:", cos)
        print("Top10:", overlap)
        print(generate(model, tok, prompt, device)[:300])

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
