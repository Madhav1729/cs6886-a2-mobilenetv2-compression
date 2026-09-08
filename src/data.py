"""CIFAR-10 input pipeline for MobileNet-v2 fine-tuning from ImageNet weights.

Images are upsampled to `resolution` (default 160) since MobileNet-v2
downsamples 32x and a native 32x32 input would collapse before the last stage.
Crop/flip run at native resolution first, then the result is upsampled -- same
effect as augmenting after resize, cheaper to compute."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .utils import seed_worker

# Statistics of the ImageNet-1k trainset (what MobileNet_V2_Weights.IMAGENET1K_V1 used).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CIFAR-10's own statistics, kept for the train-from-scratch ablation.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_transforms(resolution: int = 160, norm: str = "imagenet"):
    """Return (train_transform, test_transform).

    Train: RandomCrop(32, padding=4) -> RandomHorizontalFlip -> Resize(res, bicubic)
    Test:  Resize(res, bicubic)
    """
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if norm == "imagenet" else (CIFAR10_MEAN, CIFAR10_STD)
    interp = transforms.InterpolationMode.BICUBIC

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(resolution, interpolation=interp, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(resolution, interpolation=interp, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


def build_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    eval_batch_size: int = 256,
    resolution: int = 160,
    norm: str = "imagenet",
    num_workers: int = 4,
    seed: int = 42,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = build_transforms(resolution, norm)

    train_set = datasets.CIFAR10(data_dir, train=True, transform=train_tf, download=download)
    test_set = datasets.CIFAR10(data_dir, train=False, transform=test_tf, download=download)

    # A dedicated generator keeps shuffling order reproducible from `seed` alone.
    gen = torch.Generator()
    gen.manual_seed(seed)

    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        worker_init_fn=seed_worker, generator=gen, **common,
    )
    test_loader = DataLoader(
        test_set, batch_size=eval_batch_size, shuffle=False, drop_last=False, **common,
    )
    return train_loader, test_loader
