"""MobileNet-v2 construction for CIFAR-10. Keeps the stock torchvision
topology and swaps only the classifier, so ImageNet weights transfer directly."""

from typing import List

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_model(
    num_classes: int = 10,
    pretrained: bool = True,
    width_mult: float = 1.0,
    dropout: float = 0.2,
) -> nn.Module:
    """Stock MobileNet-v2 with a fresh `num_classes`-way head.

    `width_mult != 1.0` changes every channel count, so no ImageNet checkpoint
    exists for it; we fall back to random init and say so loudly.
    """
    if pretrained and width_mult != 1.0:
        raise ValueError(
            f"No ImageNet weights exist for width_mult={width_mult}; "
            "pass --no-pretrained to train that variant from scratch."
        )

    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights, width_mult=width_mult, dropout=dropout)

    # classifier == Sequential(Dropout, Linear(last_channel, 1000))
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    nn.init.normal_(model.classifier[1].weight, 0, 0.01)
    nn.init.zeros_(model.classifier[1].bias)
    return model


def head_parameter_names(model: nn.Module) -> List[str]:
    return [n for n, _ in model.named_parameters() if n.startswith("classifier.")]


def param_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
) -> List[dict]:
    """Four groups: {backbone, head} x {decay, no-decay}. Skips weight decay
    on BatchNorm params and biases, per common practice for conv nets."""
    head_names = set(head_parameter_names(model))
    groups = {
        "backbone_decay": {"params": [], "lr": lr_backbone, "weight_decay": weight_decay},
        "backbone_nodecay": {"params": [], "lr": lr_backbone, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": lr_head, "weight_decay": weight_decay},
        "head_nodecay": {"params": [], "lr": lr_head, "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        where = "head" if name in head_names else "backbone"
        # ndim <= 1 catches BN weight/bias and every conv/linear bias.
        kind = "nodecay" if p.ndim <= 1 else "decay"
        groups[f"{where}_{kind}"]["params"].append(p)
    return [g for g in groups.values() if g["params"]]


@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
