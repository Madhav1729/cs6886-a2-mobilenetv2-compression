"""Post-training quantization + QAT fine-tuning (Q3/Q4 headline run).

    python scripts/run_qat.py --weight-bits 4 --activation-bits 8 --epochs 5 --wandb

Pipeline: quantize weights -> attach/calibrate activations -> QAT (BN live)
-> fold BN -> quantize once more for the deployed model. See src/compress/qat.py
for why BN stays live during training.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.compress import (ActivationQuantizer, QuantConfig, WeightQuantizer,
                          apply_masks_, compression_report, compute_masks, fold_bn_,
                          fp32_baseline_bits, qat_finetune, sparsity_report)
from src.data import build_dataloaders
from src.engine import evaluate
from src.model import build_model
from src.utils import device_auto, load_checkpoint, save_checkpoint, set_seed
from src.wandb_utils import DEFAULT_ENTITY, DEFAULT_PROJECT, init_run


def get_args(argv=None):
    p = argparse.ArgumentParser(description="QAT fine-tuning of the quantized model")
    p.add_argument("--checkpoint", default="checkpoints/baseline_last.pth")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./checkpoints")

    # compression configuration
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--sensitive-bits", type=int, default=8)
    p.add_argument("--activation-bits", type=int, default=8)
    p.add_argument("--group-size", type=int, default=0)
    p.add_argument("--weight-clip", default="mse", choices=["max", "p99.9", "p99.99", "mse"])
    p.add_argument("--act-clip", default="p99.9", choices=["minmax", "p99.9", "p99.99", "p99"])
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--layer-bits", default=None,
                   help="JSON from scripts/allocate_bits.py: per-layer bit override.")
    p.add_argument("--sparsity", type=float, default=0.0,
                   help="Fraction of weights to prune (0 = no pruning).")
    p.add_argument("--prune-scope", default="global", choices=["global", "layer"],
                   help="global self-allocates sparsity; measured far better than "
                        "uniform per-layer pruning.")
    p.add_argument("--max-layer-sparsity", type=float, default=0.95,
                   help="Cap per layer so a global threshold can't zero one entirely.")
    p.add_argument("--recalibrate-every", type=int, default=1,
                   help="Refresh activation ranges every N epochs (0 disables).")

    # fine-tuning
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Kept low: the model just needs to settle into quantization bins.")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="0 by default: decay fights the STE's job of moving weights "
                        "onto grid points.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
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
    name = args.run_name or f"qat-w{args.weight_bits}a{args.activation_bits}"
    if args.sparsity > 0:
        name += f"-sp{int(args.sparsity*100)}"
    print(f"[setup] device={device} {name}")

    ckpt_args = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("args", {})
    resolution = ckpt_args.get("resolution", 160)

    model = build_model(num_classes=10, pretrained=False,
                        width_mult=ckpt_args.get("width_mult", 1.0),
                        dropout=ckpt_args.get("dropout", 0.2))
    load_checkpoint(args.checkpoint, model)
    baseline_bits = fp32_baseline_bits(model)
    model.to(device)

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, resolution=resolution,
        norm=ckpt_args.get("norm", "imagenet"), num_workers=args.num_workers,
        seed=args.seed, download=False)
    criterion = nn.CrossEntropyLoss()

    fp32_acc = evaluate(model, test_loader, criterion, device, amp=False)["acc"]
    print(f"[baseline] FP32 test acc {fp32_acc:.2f}%")

    layer_bits = {}
    if args.layer_bits:
        import json as _json
        layer_bits = _json.load(open(args.layer_bits))
        print(f"[cfg] per-layer allocation from {args.layer_bits}: {len(layer_bits)} layers")

    cfg = QuantConfig(weight_bits=args.weight_bits, sensitive_bits=args.sensitive_bits,
                      activation_bits=args.activation_bits, group_size=args.group_size,
                      weight_clip=args.weight_clip, act_clip=args.act_clip,
                      layer_bits=layer_bits)

    # save the true FP32 weights -- QAT optimizes these, so they must not get
    # overwritten by the PTQ pass below
    fp32_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    masks = None
    if args.sparsity > 0:
        masks = compute_masks(model, args.sparsity, scope=args.prune_scope,
                              max_layer_sparsity=args.max_layer_sparsity)
        apply_masks_(model, masks)
        rep = sparsity_report(masks)
        print(f"[prune] {args.prune_scope} magnitude, target {args.sparsity:.0%} -> "
              f"achieved {rep['overall']:.1%} ({rep['pruned_params']:,}/"
              f"{rep['total_params']:,} weights), worst layer {rep['worst_layer']:.1%}")

    wq = WeightQuantizer(cfg)
    wq.apply(model)
    aq = ActivationQuantizer(cfg)
    aq.attach(model)
    # calibrate with quantized weights in place, matching deployment
    aq.calibrate(model, train_loader, device, num_batches=args.calib_batches)

    ptq_acc = evaluate(model, test_loader, criterion, device, amp=False)["acc"]
    print(f"[ptq] post-training quantized acc {ptq_acc:.2f}%")

    # restore FP32 masters; qat_finetune fake-quantizes internally each step
    model.load_state_dict(fp32_state)
    if masks is not None:
        apply_masks_(model, masks)          # masters start sparse

    run = init_run(args.wandb, project=args.wandb_project,
                   entity=args.wandb_entity or None, name=name, job_type="qat",
                   config={**vars(args), "fp32_acc": fp32_acc, "ptq_acc": ptq_acc})

    result = qat_finetune(model, cfg, train_loader, test_loader, device,
                          epochs=args.epochs, lr=args.lr,
                          weight_decay=args.weight_decay,
                          act_quantizer=aq,
                          recalibrate_every=args.recalibrate_every,
                          calib_batches=args.calib_batches,
                          masks=masks)

    # qat_finetune returns FP32 masters; fold+quantize once here to avoid
    # double-rounding
    fold_bn_(model)
    if masks is not None:
        apply_masks_(model, masks)          # folding must not resurrect pruned weights
    wq_final = WeightQuantizer(cfg)
    wq_final.apply(model)
    folded_acc = evaluate(model, test_loader, criterion, device, amp=False)["acc"]
    report = compression_report(model, wq_final, cfg, baseline_bits,
                                folded=True, use_huffman=True)

    print("\n===== FINAL =====")
    print(f"  FP32 baseline        {fp32_acc:.2f}%")
    print(f"  post-training quant  {ptq_acc:.2f}%")
    print(f"  after QAT            {result['best_acc']:.2f}%")
    print(f"  after QAT + BN fold  {folded_acc:.2f}%   <- deployed model")
    print(f"  model size           {report['model_size_mb']:.3f} MB")
    print(f"  model ratio          {report['model_compression_ratio']:.1f}x")
    print(f"  weight ratio         {report['weight_compression_ratio']:.1f}x")
    print(f"  activation ratio     {report['activation_compression_ratio']:.1f}x")
    if masks is not None:
        print(f"  sparsity             {sparsity_report(masks)['overall']:.1%}")
    print(f"  bits/weight          {report['bits_per_weight']:.2f}"
          + ("   <- effective rate (per-layer allocation)" if layer_bits else ""))
    if layer_bits:
        import collections
        dist = collections.Counter(layer_bits.values())
        print(f"  allocation           " +
              ", ".join(f"{n} layers @ {b}b" for b, n in sorted(dist.items())))

    path = os.path.join(args.out_dir, f"{name}.pth")
    save_checkpoint(path, model, args.epochs, folded_acc, vars(args))
    print(f"  saved -> {path}")

    if run is not None:
        run.summary.update({"fp32_acc": fp32_acc, "ptq_acc": ptq_acc,
                            "qat_acc": result["best_acc"],
                            "folded_acc": folded_acc,
                            "quantized_acc": folded_acc,
                            "bits_per_weight": report["bits_per_weight"],
                            "mixed_precision": bool(layer_bits),
                            "sparsity": args.sparsity,
                            "weight_quant_bits": (None if layer_bits else args.weight_bits),
                            "activation_quant_bits": args.activation_bits,
                            "overhead_scales_kb": report["overhead_scales_kb"],
                            "overhead_biases_kb": report["overhead_biases_kb"],
                            "overhead_tables_kb": report["overhead_tables_kb"],
                            "compression_ratio": report["model_compression_ratio"],
                            "weight_compression_ratio": report["weight_compression_ratio"],
                            "activation_compression_ratio": report["activation_compression_ratio"],
                            "model_size_mb": report["model_size_mb"]})
        run.finish()
    return result


if __name__ == "__main__":
    main()
