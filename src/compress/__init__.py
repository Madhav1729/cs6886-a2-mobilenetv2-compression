"""From-scratch compression for MobileNet-v2 (Q2-Q4).

Pipeline order matters:
    fold_bn_(model)              # BN absorbed into convs (free, lossless)
    WeightQuantizer(cfg).apply(model)
    aq = ActivationQuantizer(cfg); aq.attach(model); aq.calibrate(model, train_loader, dev)
    compression_report(...)      # analytical size, never os.path.getsize
"""

from .fold_bn import count_folded_away, fold_bn_
from .huffman import code_lengths, entropy_bits_per_symbol, total_encoded_bits
from .quantizer import (ActivationQuantizer, ActQuant, QuantConfig, WeightQuantizer,
                        classify_layers, dequantize_tensor, quantize_tensor)
from .pruner import apply_masks_, compute_masks, sparsity_report
from .qat import qat_finetune
from .size import compression_report, fp32_baseline_bits, model_size_bits

__all__ = [
    "fold_bn_", "count_folded_away",
    "QuantConfig", "WeightQuantizer", "ActivationQuantizer", "ActQuant",
    "classify_layers", "quantize_tensor", "dequantize_tensor",
    "code_lengths", "total_encoded_bits", "entropy_bits_per_symbol",
    "model_size_bits", "compression_report", "fp32_baseline_bits",
    "qat_finetune",
    "compute_masks", "apply_masks_", "sparsity_report",
]
