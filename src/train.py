"""Q1 baseline: fine-tune ImageNet-pretrained MobileNet-v2 on CIFAR-10 (Recipe A).

    python -m src.train --epochs 20 --resolution 160 --wandb
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from .data import build_dataloaders
from .engine import evaluate, train_one_epoch
from .model import build_model, count_parameters, param_groups
from .utils import device_auto, dump_history, save_checkpoint, set_seed
from .wandb_utils import DEFAULT_ENTITY, DEFAULT_PROJECT, init_run, log_curves


def get_args(argv=None):
    p = argparse.ArgumentParser(description="MobileNet-v2 / CIFAR-10 baseline")

    # data
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--resolution", type=int, default=160,
                   help="CIFAR images are upsampled to this size (stock MobileNet-v2 downsamples 32x).")
    p.add_argument("--norm", choices=["imagenet", "cifar10"], default="imagenet")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)

    # model
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   help="Random init instead of ImageNet weights (the scratch ablation).")

    # optimisation
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr-backbone", type=float, default=0.01)
    p.add_argument("--lr-head", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=4e-5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--no-amp", dest="amp", action="store_false")

    # bookkeeping
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--out-dir", default="./checkpoints")
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    p.add_argument("--wandb-entity", default=DEFAULT_ENTITY,
                   help="Set to '' to use your wandb default entity instead.")
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed, args.deterministic)
    device = device_auto()
    print(f"[setup] device={device} seed={args.seed} resolution={args.resolution}")

    run = init_run(
        args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.run_name,
        config=vars(args),
        job_type="train",
    )

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        resolution=args.resolution,
        norm=args.norm,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(
        num_classes=10,
        pretrained=args.pretrained,
        width_mult=args.width_mult,
        dropout=args.dropout,
    ).to(device)
    print(f"[model] mobilenet_v2 width_mult={args.width_mult} "
          f"pretrained={args.pretrained} params={count_parameters(model) / 1e6:.2f}M")

    groups = param_groups(model, args.lr_backbone, args.lr_head, args.weight_decay)
    optimizer = torch.optim.SGD(groups, momentum=args.momentum, nesterov=True)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}
    best_acc, global_step = 0.0, 0
    best_path = os.path.join(args.out_dir, f"{args.run_name}_best.pth")
    last_path = os.path.join(args.out_dir, f"{args.run_name}_last.pth")

    for epoch in range(args.epochs):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch, args.epochs, base_lrs, global_step, total_steps, warmup_steps,
            grad_clip=args.grad_clip,
        )
        global_step = tr["global_step"]
        te = evaluate(model, test_loader, criterion, device, amp=args.amp)

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["test_loss"].append(te["loss"])
        history["test_acc"].append(te["acc"])
        history["lr"].append(tr["lr"])

        is_best = te["acc"] > best_acc
        best_acc = max(best_acc, te["acc"])
        save_checkpoint(last_path, model, epoch, best_acc, vars(args))
        if is_best:
            save_checkpoint(best_path, model, epoch, best_acc, vars(args))

        print(f"[epoch {epoch + 1:3d}/{args.epochs}] "
              f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.2f} "
              f"test_loss={te['loss']:.4f} test_acc={te['acc']:.2f} "
              f"best={best_acc:.2f} lr={tr['lr']:.5f} ({time.time() - t0:.0f}s)")

        if run is not None:
            run.log({
                "epoch": epoch + 1,
                "train/loss": tr["loss"], "train/acc": tr["acc"],
                "test/loss": te["loss"], "test/acc": te["acc"],
                "lr": tr["lr"], "best/acc": best_acc,
            })

    dump_history(os.path.join(args.out_dir, f"{args.run_name}_history.json"), history)
    print(f"[done] best top-1 = {best_acc:.2f}%  ->  {best_path}")
    if run is not None:
        log_curves(run, history)  # Q1(c) loss/accuracy figure, attached to the run
        run.summary["best_top1"] = best_acc
        run.finish()
    return best_acc


if __name__ == "__main__":
    main()
