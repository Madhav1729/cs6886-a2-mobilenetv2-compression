"""High-compression run: gradual pruning + distillation + QAT, targeting 15-20x.

    python scripts/run_highcompress.py --weight-bits 3 --sparsity 0.7 --epochs 40

Differs from run_qat.py in how it trains, not in the compression format: sparsity
ramps up during training rather than being applied in one step, and the FP32
baseline supervises the student as a teacher. Both matter at these ratios --
one-shot pruning to 70% put the network at chance before fine-tuning started.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.compress import (ActivationQuantizer, QuantConfig, WeightQuantizer,
                          apply_masks_, compression_report, fold_bn_,
                          fp32_baseline_bits, sparsity_report)
from src.compress.qat_gradual import qat_gradual
from src.data import build_dataloaders
from src.engine import evaluate
from src.model import build_model
from src.utils import device_auto, load_checkpoint, save_checkpoint, set_seed
from src.wandb_utils import DEFAULT_ENTITY, DEFAULT_PROJECT, init_run


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Gradual-prune + distill + QAT")
    p.add_argument("--checkpoint", default="checkpoints/baseline_last.pth")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./checkpoints")

    p.add_argument("--weight-bits", type=int, default=3)
    p.add_argument("--sensitive-bits", type=int, default=8)
    p.add_argument("--activation-bits", type=int, default=8)
    p.add_argument("--group-size", type=int, default=0)
    p.add_argument("--weight-clip", default="mse")
    p.add_argument("--act-clip", default="p99.9")
    p.add_argument("--calib-batches", type=int, default=8)

    p.add_argument("--sparsity", type=float, default=0.7,
                   help="Final sparsity, reached gradually over training.")
    p.add_argument("--prune-every", type=int, default=100,
                   help="Recompute masks every N steps.")
    p.add_argument("--max-layer-sparsity", type=float, default=0.95)

    p.add_argument("--no-distill", dest="distill", action="store_false",
                   help="Train against hard labels only, no teacher.")
    p.add_argument("--distill-alpha", type=float, default=0.9)
    p.add_argument("--distill-temperature", type=float, default=4.0)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-3,
                   help="Higher than plain QAT: the network is being reshaped by "
                        "pruning, not just nudged onto quantization grid points.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--recalibrate-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    p.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    device = device_auto()
    name = args.run_name or (f"hc-w{args.weight_bits}a{args.activation_bits}"
                             f"-sp{int(args.sparsity * 100)}"
                             f"{'-kd' if args.distill else ''}")
    print(f"[setup] device={device} {name}")

    ckpt_args = torch.load(args.checkpoint, map_location="cpu",
                           weights_only=False).get("args", {})
    resolution = ckpt_args.get("resolution", 160)

    def load_model():
        m = build_model(num_classes=10, pretrained=False,
                        width_mult=ckpt_args.get("width_mult", 1.0),
                        dropout=ckpt_args.get("dropout", 0.2))
        load_checkpoint(args.checkpoint, m)
        return m

    model = load_model()
    baseline_bits = fp32_baseline_bits(model)
    model.to(device)

    teacher = None
    if args.distill:
        teacher = load_model().to(device).eval()
        print("[setup] teacher = FP32 baseline (frozen)")

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, resolution=resolution,
        norm=ckpt_args.get("norm", "imagenet"), num_workers=args.num_workers,
        seed=args.seed, download=False)
    criterion = nn.CrossEntropyLoss()

    fp32_acc = evaluate(model, test_loader, criterion, device, amp=False)["acc"]
    print(f"[baseline] FP32 test acc {fp32_acc:.2f}%")

    cfg = QuantConfig(weight_bits=args.weight_bits, sensitive_bits=args.sensitive_bits,
                      activation_bits=args.activation_bits, group_size=args.group_size,
                      weight_clip=args.weight_clip, act_clip=args.act_clip)

    aq = ActivationQuantizer(cfg)
    aq.attach(model)
    aq.calibrate(model, train_loader, device, num_batches=args.calib_batches)

    run = init_run(args.wandb, project=args.wandb_project,
                   entity=args.wandb_entity or None, name=name, job_type="high-compress",
                   config={**vars(args), "fp32_acc": fp32_acc})

    result = qat_gradual(
        model, cfg, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        final_sparsity=args.sparsity, prune_every=args.prune_every,
        max_layer_sparsity=args.max_layer_sparsity,
        teacher=teacher, distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        act_quantizer=aq, recalibrate_every=args.recalibrate_every,
        calib_batches=args.calib_batches)

    masks = result["masks"]
    fold_bn_(model)
    if masks:
        apply_masks_(model, masks)
    wq = WeightQuantizer(cfg)
    wq.apply(model)
    if masks:
        apply_masks_(model, masks)

    final_acc = evaluate(model, test_loader, criterion, device, amp=False)["acc"]
    report = compression_report(model, wq, cfg, baseline_bits, folded=True,
                                use_huffman=True)
    achieved_sp = sparsity_report(masks)["overall"] if masks else 0.0

    print("\n===== FINAL =====")
    print(f"  FP32 baseline        {fp32_acc:.2f}%")
    print(f"  after QAT            {result['best_acc']:.2f}%")
    print(f"  deployed (folded)    {final_acc:.2f}%")
    print(f"  sparsity             {achieved_sp:.1%}")
    print(f"  model size           {report['model_size_mb']:.3f} MB")
    print(f"  model ratio          {report['model_compression_ratio']:.1f}x")
    print(f"  weight ratio         {report['weight_compression_ratio']:.1f}x")
    print(f"  activation ratio     {report['activation_compression_ratio']:.1f}x")
    print(f"  bits/weight          {report['bits_per_weight']:.2f}")

    path = os.path.join(args.out_dir, f"{name}.pth")
    save_checkpoint(path, model, args.epochs, final_acc, vars(args))
    print(f"  saved -> {path}")

    if run is not None:
        run.summary.update({
            "fp32_acc": fp32_acc, "qat_acc": result["best_acc"],
            "quantized_acc": final_acc, "folded_acc": final_acc,
            "sparsity": achieved_sp, "distill": args.distill,
            "weight_quant_bits": args.weight_bits,
            "activation_quant_bits": args.activation_bits,
            "bits_per_weight": report["bits_per_weight"],
            "compression_ratio": report["model_compression_ratio"],
            "weight_compression_ratio": report["weight_compression_ratio"],
            "activation_compression_ratio": report["activation_compression_ratio"],
            "model_size_mb": report["model_size_mb"]})
        run.finish()
    return result


if __name__ == "__main__":
    main()
