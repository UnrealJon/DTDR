#!/usr/bin/env python
"""
MISTRAL-7B DEMO (Neutral + Reproducible)

Runs up to three pipelines:
A) FP16 baseline (normal HF load)
B) PKL weights (load HF arch -> overwrite weights from .pkl -> inference)
C) bitsandbytes INT8 runtime (via BitsAndBytesConfig)

Notes:
- PKL path measures steady-state inference speed after overwrite.
- It does NOT claim inference is performed directly in compressed form.
"""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB_CONFIG = True
except Exception:
    _HAS_BNB_CONFIG = False


# ---------------------------------------------------------------------
# PKL decompression (your established blockwise format)
# ---------------------------------------------------------------------
def decompress_blockwise(payload: dict) -> torch.Tensor:
    blocks = payload["data"]
    metadata = payload["metadata"]
    bits = int(payload["bits"])
    levels = 2 ** bits

    out = []
    for q, (tmin, tmax, shape, *_rest) in zip(blocks, metadata):
        scaled = q.astype(np.float32) / (levels - 1)
        deq = scaled * (tmax - tmin) + tmin
        out.append(deq.reshape(shape))

    return torch.from_numpy(np.vstack(out)).to(torch.float16)


def load_pkl(p):
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError("PKL must contain a dict")
    return obj


def pkl_histogram(pkl):
    ctr = Counter()
    for v in pkl.values():
        if isinstance(v, dict):
            ctr[v.get("type", "<missing>")] += 1
    return dict(ctr)


def overwrite_from_pkl(model, pkl, verbose_every=50):
    t0 = time.time()
    replaced = missing = 0

    for i, (name, param) in enumerate(model.named_parameters(), 1):
        if name not in pkl:
            missing += 1
            continue

        entry = pkl[name]
        if entry["type"] == "compressed_blockwise":
            w = decompress_blockwise(entry["data"])
        elif entry["type"] == "uncompressed":
            w = torch.from_numpy(entry["data"]).to(torch.float16)
        else:
            raise ValueError(f"Unknown type {entry['type']}")

        with torch.no_grad():
            param.data = w.to(param.device)

        replaced += 1
        if verbose_every and replaced % verbose_every == 0:
            print(f"  Progress: {replaced}/291 parameters overwritten...")

    return replaced, missing, time.time() - t0


# ---------------------------------------------------------------------
# Token + timing helpers
# ---------------------------------------------------------------------
def ensure_pad_token(tok):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@torch.inference_mode()
def timed_generate(model, tok, prompt, device, max_new, do_sample):
    inputs = tok(prompt, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
    )

    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    return tok.decode(out[0], skip_special_tokens=True), dt


def ms_per_token(seconds, tokens):
    return (seconds / tokens) * 1000 if tokens else float("nan")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--cache-dir", default="./hf_cache")
    ap.add_argument("--pkl-path", default="compressed_mistral_7b.pkl")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--attn-impl", default="eager",
                    choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--out", default="demo_outputs_new.json")

    ap.add_argument("--run-fp16", action="store_true")
    ap.add_argument("--run-pkl", action="store_true")
    ap.add_argument("--run-int8-bnb", action="store_true")

    args = ap.parse_args()

    if not (args.run_fp16 or args.run_pkl or args.run_int8_bnb):
        args.run_fp16 = args.run_pkl = args.run_int8_bnb = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, humanity will",
        "The most important invention of the 21st century",
    ]

    print("=" * 78)
    print("MISTRAL-7B DEMO (Neutral + Reproducible)")
    print("=" * 78)
    print(f"Device: {device.upper()}")
    print(f"Model:  {args.model}")
    print(f"Cache:  {args.cache_dir}")
    print(f"attn_implementation: {args.attn_impl}")
    print()

    tok = ensure_pad_token(
        AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    )

    results = {"meta": vars(args), "runs": {}}

    # FP16
    if args.run_fp16:
        print("\nRUN A: FP16 baseline")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation=args.attn_impl,
            cache_dir=args.cache_dir,
        )
        load_s = time.time() - t0
        total = sum(timed_generate(model, tok, p, device,
                                   args.max_new_tokens, args.do_sample)[1]
                    for p in prompts)
        ms = ms_per_token(total, len(prompts) * args.max_new_tokens)
        print(f"✓ FP16 ms/token: {ms:.1f}")
        results["runs"]["fp16"] = {"load_s": load_s, "ms_tok": ms}
        del model
        torch.cuda.empty_cache()

    # PKL
    if args.run_pkl:
        print("\nRUN B: PKL overwrite path")
        pkl = load_pkl(args.pkl_path)
        print("PKL histogram:", pkl_histogram(pkl))

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation=args.attn_impl,
            cache_dir=args.cache_dir,
        )
        overwrite_from_pkl(model, pkl)
        total = sum(timed_generate(model, tok, p, device,
                                   args.max_new_tokens, args.do_sample)[1]
                    for p in prompts)
        ms = ms_per_token(total, len(prompts) * args.max_new_tokens)
        print(f"✓ PKL ms/token: {ms:.1f}")
        results["runs"]["pkl"] = {"ms_tok": ms}
        del model
        torch.cuda.empty_cache()

    # INT8 (bnb)
    if args.run_int8_bnb and device == "cuda" and _HAS_BNB_CONFIG:
        print("\nRUN C: bitsandbytes INT8")
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            quantization_config=bnb,
            attn_implementation=args.attn_impl,
            cache_dir=args.cache_dir,
        )
        total = sum(timed_generate(model, tok, p, device,
                                   args.max_new_tokens, args.do_sample)[1]
                    for p in prompts)
        ms = ms_per_token(total, len(prompts) * args.max_new_tokens)
        print(f"✓ bnb INT8 ms/token: {ms:.1f}")
        results["runs"]["bnb_int8"] = {"ms_tok": ms}
        del model
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDONE")
    print(f"Report written to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
