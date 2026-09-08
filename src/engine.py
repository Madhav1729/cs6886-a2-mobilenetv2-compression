"""Train / evaluate loops, shared by the baseline run and the compression sweep."""

import math
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import AverageMeter, top1_accuracy


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
    """Linear warmup then cosine decay, evaluated per optimizer step."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, base_lrs, step: int, total_steps: int, warmup_steps: int) -> float:
    """Scale every param group by the shared schedule, preserving per-group base LRs."""
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = cosine_lr(step, total_steps, warmup_steps, base_lr)
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    total_epochs: int,
    base_lrs,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    grad_clip: float = 0.0,
) -> dict:
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    use_amp = scaler is not None and scaler.is_enabled()

    pbar = tqdm(loader, desc=f"train {epoch + 1}/{total_epochs}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        lr = set_lr(optimizer, base_lrs, global_step, total_steps, warmup_steps)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = targets.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(top1_accuracy(logits.detach(), targets), bs)
        global_step += 1
        pbar.set_postfix(loss=f"{loss_m.avg:.3f}", acc=f"{acc_m.avg:.2f}", lr=f"{lr:.4f}")

    return {"loss": loss_m.avg, "acc": acc_m.avg, "lr": lr, "global_step": global_step}


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, amp: bool = True) -> dict:
    """Top-1 / loss over a full loader. Used for the baseline *and* every
    quantized configuration in the Q3 sweep."""
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    use_amp = amp and device.type == "cuda"

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        bs = targets.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(top1_accuracy(logits.float(), targets), bs)

    return {"loss": loss_m.avg, "acc": acc_m.avg}
