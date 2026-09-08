"""Standalone evaluation of a saved checkpoint (the uncompressed Q1 number).

    python -m src.eval --checkpoint checkpoints/baseline_last.pth
"""

import argparse

import torch
import torch.nn as nn

from .data import build_dataloaders
from .engine import evaluate
from .model import build_model, count_parameters
from .utils import device_auto, load_checkpoint, set_seed


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a MobileNet-v2 CIFAR-10 checkpoint")
    p.add_argument("--checkpoint", default="./checkpoints/baseline_last.pth")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--resolution", type=int, default=None,
                   help="Defaults to the resolution stored in the checkpoint's args.")
    p.add_argument("--norm", default=None)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    device = device_auto()

    ckpt_args = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("args", {})
    resolution = args.resolution or ckpt_args.get("resolution", 160)
    norm = args.norm or ckpt_args.get("norm", "imagenet")

    model = build_model(
        num_classes=10,
        pretrained=False,  # weights come from the checkpoint
        width_mult=ckpt_args.get("width_mult", 1.0),
        dropout=ckpt_args.get("dropout", 0.2),
    )
    ckpt = load_checkpoint(args.checkpoint, model)
    model.to(device)

    _, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        eval_batch_size=args.eval_batch_size,
        resolution=resolution,
        norm=norm,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, amp=args.amp)
    print(f"checkpoint : {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    print(f"resolution : {resolution}   params: {count_parameters(model) / 1e6:.2f}M")
    print(f"test loss  : {metrics['loss']:.4f}")
    print(f"test top-1 : {metrics['acc']:.2f}%")
    return metrics


if __name__ == "__main__":
    main()
