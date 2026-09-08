"""Quantization-aware fine-tuning via straight-through estimator (STE).

Each step: quantize the weights, run forward/backward as if quantization were
the identity, restore FP32 weights, then step the optimizer on those FP32
values. BatchNorm stays live during training and is folded only afterwards --
folding first trains with no normalization and recovers much less accuracy.
Folding after is equivalent because per-channel quantization commutes with BN
folding (see report/report.pdf for the derivation and the measurement)."""

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from ..engine import evaluate
from .pruner import prune_and_quantize_
from .quantizer import QuantConfig, bits_for, classify_layers, dequantize_tensor, quantize_tensor


@torch.no_grad()
def _fake_quant_(param: torch.Tensor, bits: int, cfg: QuantConfig, mask=None) -> None:
    """Replace `param` in place with its (masked) quantize->dequantize image."""
    prune_and_quantize_(param, mask, bits, cfg)


def qat_finetune(
    model: nn.Module,
    cfg: QuantConfig,
    train_loader: Iterable,
    test_loader: Iterable,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.1,
    log_every: int = 50,
    act_quantizer=None,
    recalibrate_every: int = 1,
    calib_batches: int = 8,
    leave_quantized: bool = False,
    masks=None,
) -> Dict[str, float]:
    """Fine-tune `model` so its weights tolerate quantization. `model` should
    have activation quantizers attached/calibrated and BatchNorm still live.

    Pass `act_quantizer` to refresh activation ranges every `recalibrate_every`
    epochs, since ranges measured before training go stale as weights drift.
    Weight decay defaults to 0: it fights the STE, which is trying to settle
    weights onto quantization grid points rather than shrink them.
    """
    kinds = classify_layers(model)
    targets = {n: p for n, p in model.named_parameters()
               if n in kinds and n not in cfg.skip_layers}
    bits = {n: bits_for(kinds[n], cfg, n) for n in targets}
    if masks is not None:                       # start from a sparse master
        with torch.no_grad():
            for n, p in targets.items():
                if n in masks:
                    p.data.mul_(masks[n])
    shadow = {n: p.detach().clone() for n, p in targets.items()}

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(len(train_loader), 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best = 0.0
    metrics = {"acc": 0.0}
    for epoch in range(epochs):
        if act_quantizer is not None and recalibrate_every and epoch > 0 \
                and epoch % recalibrate_every == 0:
            # ranges are re-measured with BN in eval mode, matching deployment
            with torch.no_grad():
                for n, p in targets.items():
                    shadow[n].copy_(p.data)
                    _fake_quant_(p, bits[n], cfg, None if masks is None else masks.get(n))
                act_quantizer.recalibrate(model, train_loader, device, calib_batches)
                for n, p in targets.items():
                    p.data.copy_(shadow[n])
            print(f"[qat] recalibrated activation ranges at epoch {epoch+1}", flush=True)

        model.train()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 1) stash FP32 masters, install quantized weights
            with torch.no_grad():
                for n, p in targets.items():
                    shadow[n].copy_(p.data)
                    _fake_quant_(p, bits[n], cfg, None if masks is None else masks.get(n))

            # 2) gradients are taken at the quantized point
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()

            # 3) restore masters so the update applies to full-precision weights
            with torch.no_grad():
                for n, p in targets.items():
                    p.data.copy_(shadow[n])

            # 4) STE update
            optimizer.step()
            scheduler.step()

            # 5) re-apply the pruning mask: the optimizer would otherwise pull
            #    pruned weights off zero and silently undo the sparsity.
            if masks is not None:
                with torch.no_grad():
                    for n, p in targets.items():
                        if n in masks:
                            p.data.mul_(masks[n])

            if log_every and step % log_every == 0:
                print(f"  [qat] epoch {epoch+1}/{epochs} step {step} loss {loss.item():.4f}",
                      flush=True)

        # evaluate the model as it would actually be deployed: quantized
        with torch.no_grad():
            for n, p in targets.items():
                shadow[n].copy_(p.data)
                _fake_quant_(p, bits[n], cfg, None if masks is None else masks.get(n))
        metrics = evaluate(model, test_loader, criterion, device, amp=False)
        best = max(best, metrics["acc"])
        print(f"[qat] epoch {epoch+1}/{epochs}  quantized test acc {metrics['acc']:.2f}%  "
              f"(best {best:.2f}%)", flush=True)

        # restore masters for the next epoch of training
        with torch.no_grad():
            for n, p in targets.items():
                p.data.copy_(shadow[n])

    # leave FP32 masters in place -- quantizing here then folding BN would
    # double-quantize (fold rescales already-rounded values, then re-rounds)
    with torch.no_grad():
        for n, p in targets.items():
            if leave_quantized:
                _fake_quant_(p, bits[n], cfg, None if masks is None else masks.get(n))
            else:
                p.data.copy_(shadow[n])

    return {"best_acc": best, "final_acc": metrics["acc"]}
