"""Correctness-first reference implementations for FlashForge.

Tensor convention: [batch, heads, sequence, head_dim].
Causal semantics intentionally match FlashAttention's lower-right mask when
q_len != kv_len.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def causal_lower_right_mask(nq: int, nk: int, *, device=None) -> torch.Tensor:
    q = torch.arange(nq, device=device)[:, None]
    k = torch.arange(nk, device=device)[None, :]
    # FlashAttention >= 2.1 semantics: lower-right alignment.
    return k <= (q + nk - nq)


def _make_keep_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool,
    keep_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    b, h, nq, _ = q.shape
    nk = k.shape[-2]
    mask = torch.ones((b, h, nq, nk), dtype=torch.bool, device=q.device)
    if causal:
        mask &= causal_lower_right_mask(nq, nk, device=q.device)[None, None, :, :]
    if keep_mask is not None:
        km = keep_mask.to(device=q.device, dtype=torch.bool)
        mask &= torch.broadcast_to(km, mask.shape)
    return mask


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    keep_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Stable exact attention with zero output for fully-masked rows."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be [B, H, N, D]")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError("batch/head dimensions must match and k.shape == v.shape")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("head dimensions must match")

    d = q.shape[-1]
    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = _make_keep_mask(q, k, causal=causal, keep_mask=keep_mask)

    neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
    masked_scores = torch.where(mask, scores, neg_inf)
    row_has_any = mask.any(dim=-1, keepdim=True)
    row_max = masked_scores.max(dim=-1, keepdim=True).values
    safe_row_max = torch.where(row_has_any, row_max, torch.zeros_like(row_max))
    exp_scores = torch.where(mask, torch.exp(masked_scores - safe_row_max), torch.zeros_like(scores))
    denom = exp_scores.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 0, exp_scores / torch.clamp_min(denom, torch.finfo(scores.dtype).tiny), torch.zeros_like(exp_scores))
    return torch.matmul(probs, v)


def online_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    keep_mask: Optional[torch.Tensor] = None,
    tile_k: int = 32,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Blockwise online-softmax reference matching the fused CUDA algorithm."""
    if tile_k <= 0:
        raise ValueError("tile_k must be positive")
    bsz, heads, nq, d = q.shape
    nk = k.shape[-2]
    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    mask = _make_keep_mask(q, k, causal=causal, keep_mask=keep_mask)
    out = torch.zeros_like(q)

    for b in range(bsz):
        for h in range(heads):
            for qi in range(nq):
                m = torch.tensor(float("-inf"), dtype=q.dtype, device=q.device)
                l = torch.tensor(0.0, dtype=q.dtype, device=q.device)
                acc = torch.zeros((d,), dtype=q.dtype, device=q.device)
                for ks in range(0, nk, tile_k):
                    ke = min(ks + tile_k, nk)
                    valid = mask[b, h, qi, ks:ke]
                    if not bool(valid.any()):
                        continue
                    scores = (k[b, h, ks:ke] @ q[b, h, qi]) * scale
                    valid_scores = scores[valid]
                    tile_m = valid_scores.max()
                    m_new = torch.maximum(m, tile_m)
                    alpha = torch.tensor(0.0, dtype=q.dtype, device=q.device) if torch.isneginf(m) else torch.exp(m - m_new)
                    p = torch.where(valid, torch.exp(scores - m_new), torch.zeros_like(scores))
                    acc = acc * alpha + p @ v[b, h, ks:ke]
                    l = l * alpha + p.sum()
                    m = m_new
                if float(l) > 0.0:
                    out[b, h, qi] = acc / l
    return out


def split_kv_decode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    splits: int = 8,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """FlashDecoding-style split-KV for q_len=1 with stable partial-state merge."""
    if q.shape[-2] != 1:
        raise ValueError("split_kv_decode_reference expects q_len == 1")
    if splits <= 0:
        raise ValueError("splits must be positive")
    bsz, heads, _, d = q.shape
    nk = k.shape[-2]
    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    out = torch.zeros_like(q)

    for b in range(bsz):
        for h in range(heads):
            partials = []
            for s in range(splits):
                ks = (s * nk) // splits
                ke = ((s + 1) * nk) // splits
                if ks == ke:
                    continue
                scores = (k[b, h, ks:ke] @ q[b, h, 0]) * scale
                m = scores.max()
                p = torch.exp(scores - m)
                l = p.sum()
                o = p @ v[b, h, ks:ke]
                partials.append((m, l, o))
            if not partials:
                continue
            global_m = torch.stack([p[0] for p in partials]).max()
            denom = torch.zeros((), dtype=q.dtype, device=q.device)
            numer = torch.zeros((d,), dtype=q.dtype, device=q.device)
            for m, l, o in partials:
                w = torch.exp(m - global_m)
                denom = denom + l * w
                numer = numer + o * w
            out[b, h, 0] = numer / denom
    return out
