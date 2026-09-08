"""Seeding, metric tracking and checkpoint helpers shared by train/eval/compress."""

import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG this project touches. `deterministic=True` also forces
    cuDNN reproducible mode, off by default since it costs speed for little gain."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init_fn so augmentation is reproducible across workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AverageMeter:
    """Running mean of a scalar (loss, accuracy, ...)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy in percent for one batch."""
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100.0


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str, model, epoch: int, best_acc: float, args_dict: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "args": args_dict,
        },
        path,
    )


def load_checkpoint(path: str, model, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt


def dump_history(path: str, history: dict) -> None:
    """Persist per-epoch curves so Q1(c) plots can be regenerated without wandb."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
