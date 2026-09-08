"""Q3 sweep: run the compression pipeline at several bit-widths/sparsity levels
and log one wandb run per config, for the Parallel Coordinates chart.

    python scripts/sweep_compression.py \
        --checkpoint checkpoints/baseline_last.pth \
        --weight-bits 8 6 4 --activation-bits 8 6 --sparsity 0 0.5 --wandb

Each run logs weight/activation/sensitive bits, sparsity, and the resulting
compression ratio, size, and accuracy. Activation ranges are calibrated on
training batches so the reported test accuracy isn't touched by test data.
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.compress import (ActivationQuantizer, QuantConfig, WeightQuantizer,
                          apply_masks_, compression_report, compute_masks, fold_bn_,
                          fp32_baseline_bits, sparsity_report)
from src.data import build_dataloaders
from src.engine import evaluate
from src.model import build_model
from src.utils import device_auto, load_checkpoint, set_seed
from src.wandb_utils import DEFAULT_ENTITY, DEFAULT_PROJECT, init_run, log_sweep_table


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Q3 compression sweep")
    p.add_argument("--checkpoint", default="checkpoints/baseline_last.pth")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--weight-bits", type=int, nargs="+", default=[8, 6, 4])
    p.add_argument("--activation-bits", type=int, nargs="+", default=[8, 6])
    p.add_argument("--sensitive-bits", type=int, nargs="+", default=[8],
                   help="Bits for depthwise / stem / classifier (the ~5%% of params "
                        "that are disproportionately quantization-sensitive). Accepts "
                        "several values to sweep the mixed-precision knob itself.")
    p.add_argument("--sparsity", type=float, nargs="+", default=[0.0],
                   help="Fraction of pointwise-conv weights pruned before quantization "
                        "(0 = no pruning). Pruned weights become quantization code 0, "
                        "which Huffman codes cheaply, so sparsity adds no separate "
                        "index/bitmap storage.")
    p.add_argument("--max-layer-sparsity", type=float, default=0.95)
    p.add_argument("--group-size", type=int, default=0,
                   help="Weights per scale; 0 = one scale per output channel.")
    p.add_argument("--weight-clip", default="mse", choices=["max", "p99.9", "p99.99", "mse"])
    p.add_argument("--act-clip", default="p99.9", choices=["minmax", "p99.9", "p99.99", "p99"])
    p.add_argument("--no-fold-bn", dest="fold_bn", action="store_false")
    p.add_argument("--no-huffman", dest="huffman", action="store_false")
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit-eval", type=int, default=0,
                   help="Evaluate on only this many test images (0 = full test set).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    p.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    p.add_argument("--sweep-group", default="q3-compression-sweep")
    return p.parse_args(argv)


def load_baseline(checkpoint, device):
    ckpt_args = torch.load(checkpoint, map_location="cpu", weights_only=False).get("args", {})
    model = build_model(num_classes=10, pretrained=False,
                        width_mult=ckpt_args.get("width_mult", 1.0),
                        dropout=ckpt_args.get("dropout", 0.2))
    load_checkpoint(checkpoint, model)
    return model.to(device), ckpt_args


def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    device = device_auto()

    base_model, ckpt_args = load_baseline(args.checkpoint, device)
    resolution = ckpt_args.get("resolution", 160)
    norm = ckpt_args.get("norm", "imagenet")
    baseline_bits = fp32_baseline_bits(base_model)
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size, resolution=resolution, norm=norm,
        num_workers=args.num_workers, seed=args.seed, download=False)
    criterion = nn.CrossEntropyLoss()

    columns = ["weight_quant_bits", "activation_quant_bits", "sensitive_bits", "sparsity",
               "compression_ratio",
               "weight_compression_ratio", "activation_compression_ratio",
               "model_size_mb", "bits_per_weight", "quantized_acc"]
    rows = []

    print(f"baseline: {baseline_bits/8/1024**2:.2f} MB FP32 | fold_bn={args.fold_bn} "
          f"| group={args.group_size} | wclip={args.weight_clip} aclip={args.act_clip}\n")

    for w_bits, a_bits, s_bits, sparsity in itertools.product(
            args.weight_bits, args.activation_bits, args.sensitive_bits, args.sparsity):
        set_seed(args.seed)  # identical calibration batches for every config
        cfg = QuantConfig(weight_bits=w_bits, sensitive_bits=s_bits,
                          activation_bits=a_bits, group_size=args.group_size,
                          weight_clip=args.weight_clip, act_clip=args.act_clip)

        # fresh copy so each configuration starts from the same baseline weights
        model = build_model(num_classes=10, pretrained=False,
                            width_mult=ckpt_args.get("width_mult", 1.0),
                            dropout=ckpt_args.get("dropout", 0.2))
        model.load_state_dict(base_state)
        model.to(device).eval()

        masks = None
        if sparsity > 0:
            masks = compute_masks(model, sparsity, scope="global",
                                  max_layer_sparsity=args.max_layer_sparsity)
            apply_masks_(model, masks)

        if args.fold_bn:
            fold_bn_(model)
            if masks is not None:
                apply_masks_(model, masks)

        wq = WeightQuantizer(cfg)
        wq.apply(model)
        if masks is not None:
            apply_masks_(model, masks)

        aq = ActivationQuantizer(cfg)
        aq.attach(model)
        aq.calibrate(model, train_loader, device, num_batches=args.calib_batches)

        metrics = evaluate(model, test_loader, criterion, device, amp=False)
        report = compression_report(model, wq, cfg, baseline_bits,
                                    folded=args.fold_bn, use_huffman=args.huffman)

        sp_tag = f" sp{int(sparsity*100)}" if sparsity > 0 else ""
        print(f"[w{w_bits} a{a_bits} s{s_bits}{sp_tag}] acc={metrics['acc']:.2f}%  "
              f"model={report['model_compression_ratio']:.1f}x  "
              f"weights={report['weight_compression_ratio']:.1f}x  "
              f"acts={report['activation_compression_ratio']:.1f}x  "
              f"size={report['model_size_mb']:.3f}MB  "
              f"b/w={report['bits_per_weight']:.2f}", flush=True)

        run = init_run(args.wandb, project=args.wandb_project,
                       entity=args.wandb_entity or None,
                       name=f"w{w_bits}a{a_bits}s{s_bits}" + (f"sp{int(sparsity*100)}" if sparsity > 0 else ""),
                       group=args.sweep_group, job_type="compression-sweep",
                       config={"weight_quant_bits": w_bits,
                               "activation_quant_bits": a_bits,
                               "sensitive_bits": s_bits,
                               "sparsity": sparsity,
                               "group_size": args.group_size,
                               "weight_clip": args.weight_clip,
                               "act_clip": args.act_clip,
                               "fold_bn": args.fold_bn,
                               "huffman": args.huffman,
                               "checkpoint": args.checkpoint,
                               "seed": args.seed})
        rows.append([w_bits, a_bits, s_bits, sparsity,
                     report["model_compression_ratio"],
                     report["weight_compression_ratio"],
                     report["activation_compression_ratio"],
                     report["model_size_mb"],
                     report["bits_per_weight"],
                     metrics["acc"]])
        if run is not None:
            run.summary.update({
                "compression_ratio": report["model_compression_ratio"],
                "weight_compression_ratio": report["weight_compression_ratio"],
                "activation_compression_ratio": report["activation_compression_ratio"],
                "model_size_mb": report["model_size_mb"],
                "bits_per_weight": report["bits_per_weight"],
                "quantized_acc": metrics["acc"],
                "achieved_sparsity": sparsity_report(masks)["overall"] if masks else 0.0,
                "overhead_scales_kb": report["overhead_scales_kb"],
                "overhead_biases_kb": report["overhead_biases_kb"],
                "overhead_tables_kb": report["overhead_tables_kb"],
            })
            run.finish()

    summary_run = init_run(args.wandb, project=args.wandb_project,
                           entity=args.wandb_entity or None, name="sweep-summary",
                           group=args.sweep_group, job_type="compression-sweep-summary")
    log_sweep_table(summary_run, rows, columns)
    if summary_run is not None:
        summary_run.finish()
    return rows


if __name__ == "__main__":
    main()
