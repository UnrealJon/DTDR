#!/usr/bin/env python3
"""
DTDR-domain "mini-transformer" proof-of-concept (one-sided Hadamard DTDR)

What this demonstrates:
- We store each Linear layer as DTDR weights: W_tilde = W @ H  (right-side Hadamard)
- During forward pass we DO NOT reconstruct W.
- We compute y = (H x) @ W_tilde^T   (equivalent to y = x @ W^T, up to quantization error)

This is a feasibility demo, not an optimized kernel:
- We dequantize weights to float on each forward. A real implementation would fuse:
  Hadamard(x) + int8-weight dot + scaling + accumulation, without materializing float weights.
- Softmax / LayerNorm remain in FP (as in most quantized inference pipelines).

Run:
  python dtdr_mini_transformer_poc.py
Optional:
  CUDA=1 python dtdr_mini_transformer_poc.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Fast Hadamard transform
# -----------------------------
def hadamard(x: torch.Tensor) -> torch.Tensor:
    """
    Orthonormal Hadamard transform along the last dimension.
    Works for any prefix shape, last dimension must be power of two.
    """
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard requires power-of-two length, got {n}")

    orig_shape = x.shape
    y = x.reshape(-1, n).clone()  # flatten prefix dims

    h = 1
    while h < n:
        # reshape into blocks of size 2h
        y = y.view(-1, n // (2 * h), 2 * h)
        left = y[..., :h]
        right = y[..., h:2 * h]
        y = torch.cat([left + right, left - right], dim=-1)
        y = y.view(-1, n)
        h *= 2

    y = y.view(*orig_shape)
    y = y / math.sqrt(n)  # orthonormal scaling
    return y

# -----------------------------
# Simple per-row INT8 quantizer
# -----------------------------
def quantize_int8_per_row(W: torch.Tensor):
    """
    Symmetric per-row INT8 quantization.
    Returns:
      Q: int8 tensor same shape as W
      s: float scale tensor shape (out, 1)
    """
    eps = 1e-8
    max_abs = W.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    s = max_abs / 127.0
    Q = torch.round(W / s).clamp(-127, 127).to(torch.int8)
    return Q, s


def dequant_int8_per_row(Q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # Q is int8, s is float (out,1) broadcastable
    return Q.float() * s


# -----------------------------
# DTDR Linear: W_tilde = W @ H
# Forward: y = (H x) @ W_tilde^T
# -----------------------------
class DTDRLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Stored DTDR-INT8 weights + per-row scale
        self.register_buffer("Q_tilde", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("s_tilde", torch.empty(out_features, 1, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def from_fp_weight(self, W: torch.Tensor, b: torch.Tensor | None = None):
        """
        Initialize from a floating weight W of shape (out, in).
        DTDR: W_tilde = W @ H  (right-side Hadamard mixing input dimension).
        Practical implementation: apply Hadamard to each row of W.
        """
        if W.shape != (self.out_features, self.in_features):
            raise ValueError(f"W shape {W.shape} mismatch expected {(self.out_features, self.in_features)}")

        # Compute W_tilde = W H via Hadamard on row vectors (right multiply by H)
        W_tilde = hadamard(W.float())

        Q, s = quantize_int8_per_row(W_tilde)
        self.Q_tilde.copy_(Q)
        self.s_tilde.copy_(s)

        if self.use_bias and b is not None:
            self.bias.data.copy_(b.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        Compute y = (H x) @ dequant(W_tilde)^T + bias
        """
        # Hadamard on activations (this is the "DTDR-domain inference" ingredient)
        xh = hadamard(x.float())

        # PoC: dequantize weights every forward (not optimized!)
        Wt = dequant_int8_per_row(self.Q_tilde, self.s_tilde)  # (out, in)

        y = xh @ Wt.t()
        if self.use_bias:
            y = y + self.bias
        return y


# -----------------------------
# Mini Transformer block
# -----------------------------
class BaselineMiniTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,D)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,T,T)
        p = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = p @ v  # (B,H,T,D)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        h = self.ln1(x)
        x = x + self.attn(h)

        # MLP block
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class DTDRMiniTransformer(nn.Module):
    """
    Same topology, but all Linear layers are DTDRLinear using one-sided Hadamard DTDR weights.
    LN + softmax remain standard FP.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.q = DTDRLinear(d_model, d_model, bias=False)
        self.k = DTDRLinear(d_model, d_model, bias=False)
        self.v = DTDRLinear(d_model, d_model, bias=False)
        self.o = DTDRLinear(d_model, d_model, bias=False)

        self.fc1 = DTDRLinear(d_model, d_ff, bias=True)
        self.fc2 = DTDRLinear(d_ff, d_model, bias=True)

    @torch.no_grad()
    def load_from_baseline(self, base: BaselineMiniTransformer):
        # Copy LN parameters exactly
        self.ln1.load_state_dict(base.ln1.state_dict())
        self.ln2.load_state_dict(base.ln2.state_dict())

        # Convert each FP weight to DTDR-INT8 (no reconstruction needed at runtime)
        self.q.from_fp_weight(base.q.weight.data, None)
        self.k.from_fp_weight(base.k.weight.data, None)
        self.v.from_fp_weight(base.v.weight.data, None)
        self.o.from_fp_weight(base.o.weight.data, None)

        self.fc1.from_fp_weight(base.fc1.weight.data, base.fc1.bias.data)
        self.fc2.from_fp_weight(base.fc2.weight.data, base.fc2.bias.data)

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        p = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = p @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h)

        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


# -----------------------------
# Metrics
# -----------------------------
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def main():
    use_cuda = os.environ.get("CUDA", "0") == "1" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(0)

    # Choose power-of-two d_model for Hadamard convenience in this PoC
    d_model = 256
    n_heads = 4
    d_ff = 1024

    B = 2
    T = 64

    # Baseline FP model
    base = BaselineMiniTransformer(d_model, n_heads, d_ff).to(device).eval()

    # DTDR-domain model (INT8 DTDR weights, Hadamard on activations)
    dtdr = DTDRMiniTransformer(d_model, n_heads, d_ff).to(device).eval()
    dtdr.load_from_baseline(base)

    # Random "token" activations (like post-embedding states)
    x = torch.randn(B, T, d_model, device=device, dtype=torch.float32)


    with torch.no_grad():
        y_base = base(x).float()
        y_dtdr = dtdr(x).float()

    print("Device:", device)
    print("Shapes:", x.shape, y_base.shape, y_dtdr.shape)
    print("Cosine similarity (DTDR vs FP):", cosine_similarity(y_dtdr, y_base))
    print("Relative L2 error (DTDR vs FP):", rel_l2(y_dtdr, y_base))

    # Optional: demonstrate graceful degradation under DTDR-domain corruption
    # (This is *not* necessary for feasibility, but shows the "distributed" flavor.)
    with torch.no_grad():
        # Flip a small fraction of bytes in one tensor as a toy corruption
        frac = 0.001
        Q = dtdr.q.Q_tilde
        n = Q.numel()
        k = max(1, int(frac * n))
        idx = torch.randperm(n, device=Q.device)[:k]
        flat = Q.view(-1)
        flat[idx] = flat[idx] ^ 0x7F  # xor-flip bits
        y_corrupt = dtdr(x).float()

    print(f"After toy corruption in DTDR weights (frac={frac:.4f}):")
    print("  Cosine similarity (corrupt vs FP):", cosine_similarity(y_corrupt, y_base))
    print("  Relative L2 error (corrupt vs FP):", rel_l2(y_corrupt, y_base))


if __name__ == "__main__":
    main()
