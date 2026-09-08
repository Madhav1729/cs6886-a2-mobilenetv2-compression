"""Per-layer quantization sensitivity profile, for mixed-precision allocation.

Quantizes one layer at a time (everything else stays FP32) and records the
accuracy drop, isolating each layer's own contribution.

    python scripts/sensitivity.py --bits 3 --out sensitivity_3bit.json

Use this to RANK layers, not to predict a config's joint accuracy -- individual
drops don't add up linearly once many layers are quantized together.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.compress import QuantConfig, fold_bn_
from src.compress.quantizer import classify_layers, dequantize_tensor, quantize_tensor
from src.data import build_dataloaders
from src.engine import evaluate
from src.model import build_model
from src.utils import device_auto, load_checkpoint, set_seed


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Per-layer quantization sensitivity")
    p.add_argument("--checkpoint", default="checkpoints/baseline_last.pth")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--bits", type=int, default=3, help="Bit-width to probe each layer at.")
    p.add_argument("--group-size", type=int, default=0)
    p.add_argument("--weight-clip", default="mse")
    p.add_argument("--limit-eval", type=int, default=2000,
                   help="Test images per evaluation (0 = full test set). 2000 gives "
                        "~+-0.9%% resolution, enough to rank layers.")
    p.add_argument("--eval-batch-size", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="sensitivity.json")
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    device = device_auto()

    ckpt_args = torch.load(args.checkpoint, map_location="cpu",
                           weights_only=False).get("args", {})

    def fresh():
        m = build_model(num_classes=10, pretrained=False,
                        width_mult=ckpt_args.get("width_mult", 1.0),
                        dropout=ckpt_args.get("dropout", 0.2))
        load_checkpoint(args.checkpoint, m)
        return fold_bn_(m.to(device).eval())

    _, test_loader = build_dataloaders(
        data_dir=args.data_dir, eval_batch_size=args.eval_batch_size,
        resolution=ckpt_args.get("resolution", 160), norm=ckpt_args.get("norm", "imagenet"),
        num_workers=args.num_workers, seed=args.seed, download=False)

    if args.limit_eval:
        xs, ys = [], []
        for x, y in test_loader:
            xs.append(x); ys.append(y)
            if sum(t.shape[0] for t in xs) >= args.limit_eval:
                break
        X = torch.cat(xs)[:args.limit_eval].to(device)
        Y = torch.cat(ys)[:args.limit_eval].to(device)

        @torch.no_grad()
        def accuracy(model):
            model.eval(); correct = 0
            for i in range(0, X.shape[0], args.eval_batch_size):
                xb = X[i:i + args.eval_batch_size]
                correct += (model(xb).argmax(1) == Y[i:i + args.eval_batch_size]).sum().item()
            return 100.0 * correct / X.shape[0]
    else:
        criterion = nn.CrossEntropyLoss()

        def accuracy(model):
            return evaluate(model, test_loader, criterion, device, amp=False)["acc"]

    cfg = QuantConfig(weight_bits=args.bits, sensitive_bits=args.bits,
                      group_size=args.group_size, weight_clip=args.weight_clip)

    base = fresh()
    reference = accuracy(base)
    kinds = classify_layers(base)
    sizes = {n: base.state_dict()[n].numel() for n in kinds}
    print(f"reference (folded FP32): {reference:.2f}%  |  {len(kinds)} layers "
          f"probed at {args.bits}-bit\n", flush=True)
    print(f"{'layer':<36}{'kind':<12}{'params':>10}{'acc':>8}{'drop':>8}")

    records = []
    for name, kind in kinds.items():
        model = fresh()
        state = model.state_dict()
        w = state[name].float()
        codes, scales, n_real = quantize_tensor(w, args.bits, cfg)
        state[name] = dequantize_tensor(codes, scales, w.shape, n_real)
        model.load_state_dict(state)

        acc = accuracy(model)
        drop = reference - acc
        records.append({"layer": name, "kind": kind, "params": sizes[name],
                        "acc": acc, "drop": drop})
        print(f"{name[:35]:<36}{kind:<12}{sizes[name]:>10,}{acc:>7.2f}%{drop:>7.2f}",
              flush=True)

    records.sort(key=lambda r: -r["drop"])
    payload = {"reference_acc": reference, "probe_bits": args.bits,
               "total_params": sum(sizes.values()), "layers": records}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    tolerant = [r for r in records if r["drop"] < 0.5]
    print(f"\nmost sensitive : {records[0]['layer']} "
          f"({records[0]['params']:,} params, {records[0]['drop']:.2f} drop)")
    print(f"tolerant (<0.5): {len(tolerant)}/{len(records)} layers, "
          f"{sum(r['params'] for r in tolerant):,} params "
          f"({100 * sum(r['params'] for r in tolerant) / sum(sizes.values()):.1f}% of weights)")
    print(f"wrote {args.out}")
    return payload


if __name__ == "__main__":
    main()
