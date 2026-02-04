#!/usr/bin/env python3
"""
Experiment 03+ : DTDR graceful degradation under controlled corruption

This script is fully self-contained.

It demonstrates:
- Distributed robustness of DTDR parameter storage
- Smooth, monotonic degradation with increasing corruption
- Evaluation on model parameters (not embeddings)
"""

import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Hadamard transform
# ============================================================

def hadamard(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError("Hadamard requires power-of-two length")

    y = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2 * h)
        a, b = y[..., :h], y[..., h:]
        y = torch.cat([a + b, a - b], dim=-1)
        y = y.view(-1, n)
        h *= 2

    return y.view(x.shape) / math.sqrt(n)


# ============================================================
# INT8 quantisation
# ============================================================

def quantize_int8_per_row(W):
    eps = 1e-8
    max_abs = W.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    s = max_abs / 127.0
    Q = torch.round(W / s).clamp(-127, 127).to(torch.int8)
    return Q, s


def dequant_int8_per_row(Q, s):
    return Q.float() * s


# ============================================================
# DTDR Linear layer
# ============================================================

class DTDRLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.register_buffer("Q", torch.empty(out_f, in_f, dtype=torch.int8))
        self.register_buffer("s", torch.empty(out_f, 1))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    @torch.no_grad()
    def from_fp(self, W, b=None):
        Wt = hadamard(W.float())
        Q, s = quantize_int8_per_row(Wt)
        self.Q.copy_(Q)
        self.s.copy_(s)
        if self.bias is not None and b is not None:
            self.bias.copy_(b.float())

    def forward(self, x):
        xh = hadamard(x.float())
        Wt = dequant_int8_per_row(self.Q, self.s)
        y = xh @ Wt.t()
        return y + self.bias if self.bias is not None else y


# ============================================================
# Minimal transformer
# ============================================================

class MiniTransformer(nn.Module):
    def __init__(self, d_model=256, d_ff=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        h = self.ln1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        p = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(x.size(-1)), dim=-1)
        x = x + self.o(p @ v)
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class DTDRMiniTransformer(nn.Module):
    def __init__(self, d_model=256, d_ff=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.q = DTDRLinear(d_model, d_model, False)
        self.k = DTDRLinear(d_model, d_model, False)
        self.v = DTDRLinear(d_model, d_model, False)
        self.o = DTDRLinear(d_model, d_model, False)
        self.fc1 = DTDRLinear(d_model, d_ff)
        self.fc2 = DTDRLinear(d_ff, d_model)

    @torch.no_grad()
    def load_from_fp(self, base):
        self.ln1.load_state_dict(base.ln1.state_dict())
        self.ln2.load_state_dict(base.ln2.state_dict())
        self.q.from_fp(base.q.weight)
        self.k.from_fp(base.k.weight)
        self.v.from_fp(base.v.weight)
        self.o.from_fp(base.o.weight)
        self.fc1.from_fp(base.fc1.weight, base.fc1.bias)
        self.fc2.from_fp(base.fc2.weight, base.fc2.bias)

    def forward(self, x):
        h = self.ln1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        p = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(x.size(-1)), dim=-1)
        x = x + self.o(p @ v)
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


# ============================================================
# Metrics & corruption
# ============================================================

def cosine(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def rel_l2(a, b):
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def corrupt(Q, frac):
    if frac == 0.0:
        return
    n = Q.numel()
    k = max(1, int(frac * n))
    idx = torch.randperm(n, device=Q.device)[:k]
    Q.view(-1)[idx] ^= 0x7F


# ============================================================
# Experiment
# ============================================================

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = MiniTransformer().to(device).eval()
    dtdr = DTDRMiniTransformer().to(device).eval()
    dtdr.load_from_fp(base)

    x = torch.randn(2, 64, 256, device=device)

    with torch.no_grad():
        y_ref = base(x)

    levels = [0.0, 0.001, 0.005, 0.01, 0.02]
    rows = []

    for frac in levels:
        dtdr.load_from_fp(base)
        for m in dtdr.modules():
            if isinstance(m, DTDRLinear):
                corrupt(m.Q, frac)

        with torch.no_grad():
            y = dtdr(x)

        rows.append((frac, cosine(y, y_ref), rel_l2(y, y_ref)))
        print(f"{frac:6.3%} | cos={rows[-1][1]:.6f} | l2={rows[-1][2]:.6f}")

    with open("dtdr_corruption_sweep.csv", "w", newline="") as f:
        csv.writer(f).writerows(
            [("corruption_frac", "cosine_similarity", "relative_l2"), *rows]
        )


if __name__ == "__main__":
    main()
