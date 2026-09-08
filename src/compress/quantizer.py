"""Weight and activation quantization for MobileNet-v2, written from scratch
(no compression library calls). See report/report.pdf for the numbers behind
each design choice."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import InvertedResidual

# ---------------------------------------------------------------- configuration


@dataclass
class QuantConfig:
    """Everything the Q3 sweep varies."""

    weight_bits: int = 4               # bits for pointwise/expansion convs
    sensitive_bits: int = 8            # bits for depthwise / stem / classifier
    activation_bits: int = 8
    group_size: int = 0                # 0 = per output channel, >0 = grouped
    per_tensor: bool = False           # ablation only; breaks with folded BN
    weight_clip: str = "mse"           # "max" | "p99.9" | "p99.99" | "mse"
    act_clip: str = "p99.9"            # "minmax" | "p99.9" | "p99.99" | "p99"
    scale_dtype_bits: int = 16         # keep at FP16 -- scale error is a
                                       # per-channel gain error, not noise,
                                       # so FP8 scales collapse accuracy
    bias_dtype_bits: int = 16          # folded BN biases stored as FP16
    skip_layers: Tuple[str, ...] = ()  # weight names to leave in FP32
    layer_bits: Dict[str, int] = field(default_factory=dict)
    # per-layer bit override (name -> bits), overrides weight_bits/sensitive_bits.
    # produced by scripts/allocate_bits.py from a measured sensitivity profile.


SENSITIVE = ("depthwise", "first", "classifier")


# ------------------------------------------------------------------- taxonomy


def classify_layers(model: nn.Module) -> Dict[str, str]:
    """Map each weight tensor name -> {pointwise, depthwise, first, classifier}.

    Depthwise is detected structurally: groups == in_channels.
    """
    kinds: Dict[str, str] = {}
    seen_first = False
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            kinds[name + ".weight"] = "classifier"
        elif isinstance(mod, nn.Conv2d):
            if not seen_first:
                kinds[name + ".weight"] = "first"
                seen_first = True
            elif mod.groups == mod.in_channels and mod.groups > 1:
                kinds[name + ".weight"] = "depthwise"
            else:
                kinds[name + ".weight"] = "pointwise"
    return kinds


def bits_for(kind: str, cfg: QuantConfig, name: str = "") -> int:
    """Bits for one weight tensor: explicit per-layer override wins, else the
    kind-based default (sensitive layers keep more bits)."""
    if name and name in cfg.layer_bits:
        return cfg.layer_bits[name]
    return cfg.sensitive_bits if kind in SENSITIVE else cfg.weight_bits


# --------------------------------------------------------- weight quantization


def _grouped(w: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    """Reshape to (out_channels, n_groups, group_size), zero-padding the tail."""
    out = w.shape[0]
    flat = w.reshape(out, -1)
    n_real = flat.shape[1]
    if group_size <= 0:
        return flat.reshape(out, 1, -1), n_real
    pad = (-n_real) % group_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(out, pad)], dim=1)
    return flat.reshape(out, -1, group_size), n_real


def _select_scale(groups: torch.Tensor, qmax: int, clip: str) -> torch.Tensor:
    """Pick the clipping magnitude per group. max|w| wastes levels on outliers;
    percentile/MSE search trades a bit of clipping error for finer resolution."""
    if clip == "max":
        return groups.abs().amax(-1, keepdim=True)
    if clip.startswith("p"):
        q = float(clip[1:]) / 100.0
        return torch.quantile(groups.abs(), q, dim=-1, keepdim=True)
    if clip == "mse":
        best_err: Optional[torch.Tensor] = None
        best_amp: Optional[torch.Tensor] = None
        amax = groups.abs().amax(-1, keepdim=True)
        for ratio in torch.linspace(0.5, 1.0, 11):
            amp = amax * ratio
            s = amp.clamp_min(1e-12) / qmax
            err = ((torch.clamp(torch.round(groups / s), -qmax - 1, qmax) * s - groups) ** 2)
            err = err.sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_amp = err, amp
            else:
                best_amp = torch.where(err < best_err, amp, best_amp)
                best_err = torch.minimum(err, best_err)
        return best_amp
    raise ValueError(f"unknown weight_clip: {clip}")


def quantize_tensor(w: torch.Tensor, bits: int, cfg: QuantConfig
                    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize one weight tensor.

    Returns (integer codes, scales, n_real_elements). Codes and scales together
    are exactly what a deployed model would store.
    """
    qmax = 2 ** (bits - 1) - 1
    if cfg.per_tensor:
        amp = w.abs().max().reshape(1, 1, 1)
        groups, n_real = w.reshape(1, 1, -1), w.numel()
    else:
        groups, n_real = _grouped(w, cfg.group_size)
        amp = _select_scale(groups, qmax, cfg.weight_clip)

    scales = amp.clamp_min(1e-12) / qmax
    codes = torch.clamp(torch.round(groups / scales), -qmax - 1, qmax)
    return codes, scales, n_real


def dequantize_tensor(codes: torch.Tensor, scales: torch.Tensor,
                      shape: torch.Size, n_real: int) -> torch.Tensor:
    """Inverse of `quantize_tensor`, dropping any group padding."""
    flat = (codes * scales).reshape(codes.shape[0], -1)[:, :n_real]
    return flat.reshape(shape)


class WeightQuantizer:
    """Fake-quantizes every conv/linear weight: quantize then dequantize back
    into FP32, so accuracy matches a true integer deployment while the model
    still runs under normal PyTorch. Storage size is computed separately in
    size.py, not measured from a saved file."""

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg
        self.codes: Dict[str, torch.Tensor] = {}
        self.scales: Dict[str, torch.Tensor] = {}
        self.kinds: Dict[str, str] = {}

    @torch.no_grad()
    def apply(self, model: nn.Module) -> nn.Module:
        cfg = self.cfg
        self.kinds = classify_layers(model)
        state = model.state_dict()

        for name, kind in self.kinds.items():
            if name in cfg.skip_layers:
                continue
            w = state[name].float()
            bits = bits_for(kind, cfg, name)
            codes, scales, n_real = quantize_tensor(w, bits, cfg)
            state[name] = dequantize_tensor(codes, scales, w.shape, n_real)
            # drop group padding -- it's a reshape artefact, not a real weight
            self.codes[name] = codes.reshape(codes.shape[0], -1)[:, :n_real].to(torch.int32)
            self.scales[name] = scales

        model.load_state_dict(state)
        return model


# ----------------------------------------------------- activation quantization


class ActQuant(nn.Module):
    """Asymmetric per-tensor activation fake-quant. While `calibrating`, it
    subsamples values it sees; `freeze()` turns those into a fixed range."""

    MAX_SAMPLES = 20_000

    def __init__(self, bits: int, clip: str):
        super().__init__()
        self.bits, self.clip = bits, clip
        self.calibrating = True
        self._samples: List[torch.Tensor] = []
        self.lo, self.hi = 0.0, 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            flat = x.detach().flatten()
            n = min(self.MAX_SAMPLES, flat.numel())
            # sample on the same device as flat (CUDA-safe), store on CPU
            idx = torch.randint(0, flat.numel(), (n,), device=flat.device)
            self._samples.append(flat[idx].float().cpu())
            return x

        levels = 2 ** self.bits - 1
        scale = max((self.hi - self.lo) / levels, 1e-12)
        zero_point = round(-self.lo / scale)
        q = torch.clamp(torch.round(x / scale) + zero_point, 0, levels)
        return (q - zero_point) * scale

    def reset(self) -> None:
        """Re-enter calibration -- used to refresh ranges mid-QAT as weights drift."""
        self.calibrating = True
        self._samples = []

    @torch.no_grad()
    def freeze(self) -> None:
        values = torch.cat(self._samples)
        self._samples = []
        if self.clip == "minmax":
            self.lo, self.hi = values.min().item(), values.max().item()
        else:
            pct = float(self.clip[1:]) / 100.0
            tail = (1.0 - pct) / 2.0
            self.lo = torch.quantile(values, tail).item()
            self.hi = torch.quantile(values, 1.0 - tail).item()
        self.calibrating = False


class ActivationQuantizer:
    """Attaches ActQuant to every ReLU6 output and every InvertedResidual block
    output. The block-output hook matters: linear-bottleneck projections and
    residual sums have no activation module, so a ReLU-only hook would leave
    them in FP32."""

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg
        self.quantizers: List[ActQuant] = []

    def attach(self, model: nn.Module) -> nn.Module:
        def hook(module, _inputs, output):
            return module._act_quant(output)

        for mod in model.modules():
            if isinstance(mod, (InvertedResidual, nn.ReLU6)):
                q = ActQuant(self.cfg.activation_bits, self.cfg.act_clip)
                mod._act_quant = q
                mod.register_forward_hook(hook)
                self.quantizers.append(q)
        return model

    @torch.no_grad()
    def calibrate(self, model: nn.Module, loader: Iterable, device: torch.device,
                  num_batches: int = 8) -> None:
        """Collect activation ranges on training batches, then freeze them
        (keeps calibration off the test set)."""
        model.eval()
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(images.to(device))
        for q in self.quantizers:
            q.freeze()

    @torch.no_grad()
    def recalibrate(self, model: nn.Module, loader: Iterable, device: torch.device,
                    num_batches: int = 8) -> None:
        """Re-measure activation ranges against the current weights."""
        for q in self.quantizers:
            q.reset()
        self.calibrate(model, loader, device, num_batches)

    def bits_per_element(self) -> int:
        return self.cfg.activation_bits
