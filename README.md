# MobileNet-v2 on CIFAR-10 + Compression

MobileNet-v2 fine-tuned on CIFAR-10 (Q1), then compressed with a from-scratch
pruning + quantization + Huffman-coding pipeline (Q2–Q4). Design rationale and
results are in `report/report.pdf`; this file is just how to run things.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Local dev: Python 3.13, CUDA 12.8, `torch==2.10.0`, `torchvision==0.25.0`.
On the Aqua cluster (CUDA 12.4): `pip install torch==2.6.0 torchvision==0.21.0
--index-url https://download.pytorch.org/whl/cu124`. Exact pins in
`requirements.txt`.

Seed is fixed at 42 everywhere (`--seed`), covering `random`, NumPy, and
PyTorch's CPU/CUDA generators.

## Commands

```bash
# Q1 — baseline training (downloads CIFAR-10 into ./data on first run)
python -m src.train --run-name baseline --epochs 20 --seed 42 --wandb

# Q1(c) — loss/accuracy curves
python scripts/plot_curves.py --history checkpoints/baseline_history.json

# Q2/Q3 — compression sweep across bit-widths, sensitive-layer bits, sparsity
python scripts/sweep_compression.py --checkpoint checkpoints/baseline_last.pth \
    --weight-bits 8 6 4 3 --activation-bits 8 6 --sensitive-bits 8 4 \
    --sparsity 0 0.5 --wandb

# Q4 — final reported config: 4-bit weights, 8-bit sensitive layers,
# 8-bit activations, 50% pruning, then QAT
python scripts/run_qat.py --weight-bits 4 --activation-bits 8 \
    --sparsity 0.5 --epochs 20 --lr 3e-3 --wandb

# per-layer sensitivity profile + bit allocation (used to justify which
# layers get protected — see report Q2b)
python scripts/sensitivity.py --bits 3 --out sensitivity_3bit.json
python scripts/allocate_bits.py --profile sensitivity_3bit.json --out layer_bits.json
```

PBS job scripts for the same commands are in `jobs/` (`sweep.pbs`, `qat.pbs`,
`sensitivity.pbs`); set `CONDA_ENV` or `VENV_PATH` once in `jobs/_env.sh`.

## Results

Baseline: **96.34%** top-1, 8.53 MB (FP32).

| config | weight bits | sparsity | accuracy | model size | ratio |
|---|---|---|---|---|---|
| W8/A8 | 8 | 0% | 96.15% | 2.082 MB | 4.1x |
| W6/A8 | 6 | 0% | 95.75% | 1.573 MB | 5.4x |
| W4/A8 (PTQ) | 4 | 0% | 93.07% | 1.049 MB | 8.1x |
| W4/A8 + QAT | 4 | 0% | 94.97% | 1.049 MB | 8.1x |
| W3/A8 + QAT | 3 | 0% | 92.79% | 0.861 MB | 9.9x |
| **W4/A8 + 50% pruning + QAT (reported, Q4)** | 4 | 50% | **93.23%** | **0.795 MB** | **10.7x** |

Full derivation, per-layer sensitivity analysis, and the Parallel Coordinates
sweep are in `report/report.pdf`.

## Layout

```
src/            data, model, train/eval loop
src/compress/   pruning, quantization, Huffman coding, size accounting, QAT
scripts/        sweep, sensitivity analysis, bit allocation, QAT entry point
jobs/           PBS scripts for the above
report/         report.tex / report.pdf
```


