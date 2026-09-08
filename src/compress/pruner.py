"""Magnitude pruning, composed with quantization.

Pruned weights become quantization code 0, and Huffman coding already gives
frequent codes short codewords -- so sparsity needs no separate index/bitmap
and costs nothing extra to store; size.py needs no changes to support it.

Thresholding is global (across all target layers) rather than per-layer:
per-layer pruning forces every layer to give up the same fraction regardless
of whether it can spare it, which measured far worse on this checkpoint."""

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from .quantizer import QuantConfig, classify_layers


@torch.no_grad()
def compute_masks(
    model: nn.Module,
    sparsity: float,
    scope: str = "global",
    protect: Iterable[str] = (),
    max_layer_sparsity: float = 0.95,
) -> Dict[str, torch.Tensor]:
    """Boolean keep-masks (True = keep) for every conv/linear weight.

    `max_layer_sparsity` caps how much any single layer may lose -- a pure
    global threshold can otherwise prune a small layer entirely.
    """
    if not 0.0 < sparsity < 1.0:
        return {}

    kinds = classify_layers(model)
    state = model.state_dict()
    protect = set(protect)
    targets = [n for n in kinds if n not in protect]

    masks: Dict[str, torch.Tensor] = {n: torch.ones_like(state[n], dtype=torch.bool)
                                      for n in kinds}

    if scope == "global":
        pool = torch.cat([state[n].detach().abs().flatten().float() for n in targets])
        # torch.quantile caps out around 16M elements; sample when larger.
        if pool.numel() > 8_000_000:
            pool = pool[torch.randperm(pool.numel(), device=pool.device)[:8_000_000]]
        threshold = torch.quantile(pool, sparsity)
        for n in targets:
            masks[n] = state[n].detach().abs() > threshold
    else:                                   # uniform per-layer
        for n in targets:
            w = state[n].detach().abs().flatten().float()
            threshold = torch.quantile(w, sparsity)
            masks[n] = state[n].detach().abs() > threshold

    # enforce the per-layer cap so no layer is wiped out
    for n in targets:
        keep = masks[n]
        frac_pruned = 1.0 - keep.float().mean().item()
        if frac_pruned > max_layer_sparsity:
            w = state[n].detach().abs()
            k = max(int(round((1.0 - max_layer_sparsity) * w.numel())), 1)
            kth = torch.topk(w.flatten(), k, largest=True).values.min()
            masks[n] = w >= kth
    return masks


@torch.no_grad()
def apply_masks_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> nn.Module:
    """Zero the pruned weights in place."""
    state = model.state_dict()
    for name, mask in masks.items():
        state[name] = state[name] * mask
    model.load_state_dict(state)
    return model


def sparsity_report(masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Overall and worst-layer sparsity actually achieved."""
    if not masks:
        return {"overall": 0.0, "worst_layer": 0.0, "pruned_params": 0, "total_params": 0}
    total = sum(m.numel() for m in masks.values())
    kept = sum(int(m.sum()) for m in masks.values())
    per_layer = [1.0 - m.float().mean().item() for m in masks.values()]
    return {
        "overall": 1.0 - kept / total,
        "worst_layer": max(per_layer),
        "pruned_params": total - kept,
        "total_params": total,
    }


@torch.no_grad()
def prune_and_quantize_(param: torch.Tensor, mask: Optional[torch.Tensor],
                        bits: int, cfg: QuantConfig) -> None:
    """Mask then fake-quantize one parameter, in place (used by QAT)."""
    from .quantizer import dequantize_tensor, quantize_tensor
    w = param.data.float()
    if mask is not None:
        w = w * mask
    codes, scales, n_real = quantize_tensor(w, bits, cfg)
    out = dequantize_tensor(codes, scales, param.shape, n_real)
    if mask is not None:
        out = out * mask          # quantization must not resurrect pruned weights
    param.data.copy_(out)
