"""Q1(c) loss/accuracy curves from the history JSON written by src/train.py.

    python scripts/plot_curves.py --history checkpoints/baseline_history.json

This is the offline fallback -- src/train.py already logs the same figure to
wandb (via src.wandb_utils.log_curves) when run with --wandb, so you normally
only need this script to regenerate a PNG for the report.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.plotting import save_curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="checkpoints/baseline_history.json")
    p.add_argument("--out", default="checkpoints/baseline_curves.png")
    args = p.parse_args()

    with open(args.history) as f:
        h = json.load(f)

    save_curves(h, args.out)
    print(f"wrote {args.out}  (best test top-1 = {max(h['test_acc']):.2f}%)")


if __name__ == "__main__":
    main()
