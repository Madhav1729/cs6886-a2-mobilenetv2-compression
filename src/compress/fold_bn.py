"""BatchNorm folding for MobileNet-v2.

At inference BN is an affine map, so it folds into the preceding conv:
    BN(conv(x)) = conv'(x),  W' = W*f,  b' = beta + (b-mu)*f,  f = gamma/sqrt(var+eps)

This removes every BN parameter from storage and is exactly lossless in FP32.
Folding does widen the per-channel weight range, though, so it must be paired
with per-channel (or grouped) quantization -- a single per-tensor scale can't
absorb that and accuracy collapses."""

import torch
import torch.nn as nn


@torch.no_grad()
def fold_bn_(model: nn.Module) -> nn.Module:
    """Fuse every adjacent (Conv2d, BatchNorm2d) pair, in place.

    torchvision's MobileNet-v2 builds each block as
    `Sequential(Conv2d, BatchNorm2d, ReLU6)`, so walking Sequential containers
    and fusing neighbouring pairs covers the whole network. Each fused BN is
    replaced by `nn.Identity` to keep module indices stable.
    """
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        for i in range(len(module) - 1):
            conv, bn = module[i], module[i + 1]
            if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
                continue

            f = bn.weight / torch.sqrt(bn.running_var + bn.eps)   # per out-channel
            w = conv.weight
            conv.weight.data = w * f.reshape(-1, *([1] * (w.dim() - 1)))

            prev_bias = conv.bias.data if conv.bias is not None else torch.zeros_like(f)
            conv.bias = nn.Parameter(bn.bias + (prev_bias - bn.running_mean) * f)

            module[i + 1] = nn.Identity()
    return model


def count_folded_away(model: nn.Module) -> int:
    """Number of BN parameters (gamma + beta) folding removes from storage."""
    return sum(m.weight.numel() + m.bias.numel()
               for m in model.modules() if isinstance(m, nn.BatchNorm2d))
