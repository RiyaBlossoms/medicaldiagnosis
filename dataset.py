"""
Dataset loading and augmentation pipeline.
LIGHTWEIGHT DEMO VERSION
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms

from config import Config


def get_transforms(cfg: Config, split: str):

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":

        return transforms.Compose([

            transforms.Resize((cfg.img_size, cfg.img_size)),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(mean, std),

        ])

    return transforms.Compose([

        transforms.Resize((cfg.img_size, cfg.img_size)),

        transforms.ToTensor(),

        transforms.Normalize(mean, std),

    ])


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:

    root = Path(cfg.data_root) / cfg.disease

    # LOAD DATASETS
    train_ds = datasets.ImageFolder(
        root / "train",
        transform=get_transforms(cfg, "train")
    )

    val_ds = datasets.ImageFolder(
        root / "val",
        transform=get_transforms(cfg, "val")
    )

    test_ds = datasets.ImageFolder(
        root / "test",
        transform=get_transforms(cfg, "test")
    )

    print("FULL TRAIN DATASET SIZE:", len(train_ds))

    # -----------------------------------
    # LIGHTWEIGHT SUBSET FOR DEMO
    # -----------------------------------

    train_subset_size = min(200, len(train_ds))
    val_subset_size   = min(50, len(val_ds))
    test_subset_size  = min(50, len(test_ds))

    train_ds = Subset(train_ds, range(train_subset_size))
    val_ds   = Subset(val_ds, range(val_subset_size))
    test_ds  = Subset(test_ds, range(test_subset_size))

    print("USING TRAIN SUBSET:", len(train_ds))
    print("USING VAL SUBSET:", len(val_ds))

    # -----------------------------------
    # SIMPLE DATALOADERS
    # -----------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
