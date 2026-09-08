"""QAT with gradual pruning and knowledge distillation, for high compression.

Two differences from qat.py, both aimed at holding accuracy at 15-20x:

1. Gradual pruning. Sparsity ramps from 0 to the target over most of training
   on a cubic schedule, instead of being applied in one step up front. Masks are
   recomputed at each pruning step from the current weight magnitudes, and the
   FP32 masters are left unmasked, so a weight pruned early can come back if it
   grows again. One-shot pruning to 70% left the network at chance level before
   fine-tuning even started, which is a much worse place to recover from.

2. Distillation. The uncompressed FP32 model is kept as a frozen teacher and
   the student matches its softened logits, not just the hard labels. The
   teacher's output distribution carries more information per example than a
   one-hot target, which matters when the student has little capacity left.
"""

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..engine import evaluate
from .pruner import compute_masks, prune_and_quantize_
from .quantizer import QuantConfig, bits_for, classify_layers


def cubic_sparsity(step: int, total_steps: int, final_sparsity: float,
                   warmup_frac: float = 0.05, ramp_frac: float = 0.70) -> float:
    """Sparsity at a given step: flat 0, cubic ramp, then flat at the target.

    The cubic shape (Zhu & Gupta) prunes fast while the network is still
    over-parameterised and slows down near the target, where each additional
    weight removed costs more.
    """
    warmup = int(total_steps * warmup_frac)
    ramp_end = int(total_steps * ramp_frac)
    if step < warmup:
        return 0.0
    if step >= ramp_end:
        return final_sparsity
    progress = (step - warmup) / max(ramp_end - warmup, 1)
    return final_sparsity * (1.0 - (1.0 - progress) ** 3)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      labels: torch.Tensor, alpha: float = 0.9, temperature: float = 4.0,
                      label_smoothing: float = 0.0) -> torch.Tensor:
    """Weighted mix of soft-target KL against the teacher and hard-label CE.

    The T^2 factor keeps the soft-target gradients the same magnitude as the
    hard-label ones when temperature changes.
    """
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.log_softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean", log_target=True,
    ) * (temperature ** 2)
    hard = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing)
    return alpha * soft + (1.0 - alpha) * hard


def qat_gradual(
    model: nn.Module,
    cfg: QuantConfig,
    train_loader: Iterable,
    test_loader: Iterable,
    device: torch.device,
    epochs: int = 30,
    lr: float = 5e-3,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.1,
    final_sparsity: float = 0.0,
    prune_every: int = 100,
    max_layer_sparsity: float = 0.95,
    teacher: Optional[nn.Module] = None,
    distill_alpha: float = 0.9,
    distill_temperature: float = 4.0,
    act_quantizer=None,
    recalibrate_every: int = 5,
    calib_batches: int = 8,
    log_every: int = 100,
) -> Dict[str, float]:
    """Fine-tune with a gradually increasing sparsity target and optional
    distillation. Returns best/final accuracy and the final masks.

    `model` should have activation quantizers attached and BatchNorm still live.
    Masks are recomputed every `prune_every` steps from the current weights.
    """
    kinds = classify_layers(model)
    targets = {n: p for n, p in model.named_parameters()
               if n in kinds and n not in cfg.skip_layers}
    bits = {n: bits_for(kinds[n], cfg, n) for n in targets}
    shadow = {n: p.detach().clone() for n, p in targets.items()}

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    masks: Dict[str, torch.Tensor] = {}
    best, global_step = 0.0, 0
    metrics = {"acc": 0.0}

    for epoch in range(epochs):
        if act_quantizer is not None and recalibrate_every and epoch > 0 \
                and epoch % recalibrate_every == 0:
            with torch.no_grad():
                for n, p in targets.items():
                    shadow[n].copy_(p.data)
                    prune_and_quantize_(p, masks.get(n), bits[n], cfg)
                act_quantizer.recalibrate(model, train_loader, device, calib_batches)
                for n, p in targets.items():
                    p.data.copy_(shadow[n])
            print(f"[qat] recalibrated activations at epoch {epoch + 1}", flush=True)

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # recompute masks on schedule; masters stay dense so pruned weights
            # can come back if they grow again
            if final_sparsity > 0 and global_step % prune_every == 0:
                target_sp = cubic_sparsity(global_step, total_steps, final_sparsity)
                if target_sp > 0:
                    masks = compute_masks(model, target_sp, scope="global",
                                          max_layer_sparsity=max_layer_sparsity)

            with torch.no_grad():
                for n, p in targets.items():
                    shadow[n].copy_(p.data)
                    prune_and_quantize_(p, masks.get(n), bits[n], cfg)

            optimizer.zero_grad(set_to_none=True)
            student_logits = model(images)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = distillation_loss(student_logits, teacher_logits, labels,
                                         distill_alpha, distill_temperature,
                                         label_smoothing)
            else:
                loss = criterion(student_logits, labels)
            loss.backward()

            with torch.no_grad():                       # restore dense masters
                for n, p in targets.items():
                    p.data.copy_(shadow[n])

            optimizer.step()
            scheduler.step()
            global_step += 1

            if log_every and global_step % log_every == 0:
                sp = sum(int((~m).sum()) for m in masks.values()) / \
                     max(sum(m.numel() for m in masks.values()), 1) if masks else 0.0
                print(f"  [qat] epoch {epoch + 1}/{epochs} step {global_step} "
                      f"loss {loss.item():.4f} sparsity {sp:.1%}", flush=True)

        # evaluate in deployed state: masked + quantized
        with torch.no_grad():
            for n, p in targets.items():
                shadow[n].copy_(p.data)
                prune_and_quantize_(p, masks.get(n), bits[n], cfg)
        metrics = evaluate(model, test_loader, criterion, device, amp=False)
        best = max(best, metrics["acc"])
        achieved = (sum(int((~m).sum()) for m in masks.values()) /
                    max(sum(m.numel() for m in masks.values()), 1)) if masks else 0.0
        print(f"[qat] epoch {epoch + 1}/{epochs}  acc {metrics['acc']:.2f}%  "
              f"(best {best:.2f}%)  sparsity {achieved:.1%}", flush=True)
        with torch.no_grad():
            for n, p in targets.items():
                p.data.copy_(shadow[n])

    # final state: apply the last mask permanently to the masters
    with torch.no_grad():
        for n, p in targets.items():
            if n in masks:
                p.data.mul_(masks[n])

    return {"best_acc": best, "final_acc": metrics["acc"], "masks": masks}
