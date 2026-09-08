"""Analytical model-size accounting (Q2c / Q4d).

Size is computed from the storage format we designed (payload + scales +
biases + code tables), not measured from a saved file -- torch.save would add
pickle framing that says nothing about our actual format. Activations aren't
part of model size since they're transient; reported separately as a
bits-per-element ratio (Q4b)."""

from typing import Dict

import torch
import torch.nn as nn

from .huffman import encoded_bits
from .quantizer import QuantConfig, WeightQuantizer, bits_for


def weight_storage_bits(quantizer: WeightQuantizer, cfg: QuantConfig,
                        use_huffman: bool = True) -> Dict[str, float]:
    """Bits needed for the quantized weights, their scales and code tables."""
    if use_huffman:
        # per-tensor code tables -- a single shared table is cheaper in table
        # size but worse overall, since merging different layers' code
        # distributions flattens them and inflates the payload
        payload = table = 0
        for codes in quantizer.codes.values():
            p_bits, t_bits = encoded_bits(codes)
            payload += p_bits
            table += t_bits
    else:
        payload = sum(c.numel() * bits_for(quantizer.kinds[n], cfg, n)
                      for n, c in quantizer.codes.items())
        table = 0

    scale_bits = sum(s.numel() for s in quantizer.scales.values()) * cfg.scale_dtype_bits
    return {"payload": float(payload), "code_table": float(table), "scales": float(scale_bits)}


def model_size_bits(model: nn.Module, quantizer: WeightQuantizer, cfg: QuantConfig,
                    folded: bool = True, use_huffman: bool = True) -> Dict[str, float]:
    """Full storage breakdown, in bits."""
    parts = weight_storage_bits(quantizer, cfg, use_huffman)

    if folded:
        # BN folded away; each conv/linear keeps one bias per output channel.
        n_bias = sum(m.out_channels for m in model.modules() if isinstance(m, nn.Conv2d))
        n_bias += sum(m.out_features for m in model.modules() if isinstance(m, nn.Linear))
        parts["biases"] = float(n_bias * cfg.bias_dtype_bits)
        parts["batchnorm"] = 0.0
    else:
        n_bn = sum(m.weight.numel() + m.bias.numel()
                   for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        parts["batchnorm"] = float(n_bn * 32)
        parts["biases"] = 0.0

    parts["total"] = sum(parts.values())
    return parts


def fp32_baseline_bits(model: nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters()) * 32)


def compression_report(model: nn.Module, quantizer: WeightQuantizer, cfg: QuantConfig,
                       baseline_bits: float, folded: bool = True,
                       use_huffman: bool = True) -> Dict[str, float]:
    """The numbers the assignment asks for, in one dict."""
    parts = model_size_bits(model, quantizer, cfg, folded, use_huffman)

    n_weights = sum(c.numel() for c in quantizer.codes.values())
    weight_only_bits = parts["payload"] + parts["code_table"] + parts["scales"]

    return {
        "model_size_mb": parts["total"] / 8 / 1024 ** 2,
        "model_compression_ratio": baseline_bits / parts["total"],
        "weight_compression_ratio": (n_weights * 32) / weight_only_bits,
        "activation_compression_ratio": 32.0 / cfg.activation_bits,
        "bits_per_weight": weight_only_bits / n_weights,
        "overhead_scales_kb": parts["scales"] / 8 / 1024,
        "overhead_tables_kb": parts["code_table"] / 8 / 1024,
        "overhead_biases_kb": parts["biases"] / 8 / 1024,
        **{f"bits_{k}": v for k, v in parts.items()},
    }
