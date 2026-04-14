"""
Dataset and data loading for semi-supervised DeepSDF training.

Each shape has:
  - Supervised points: 3D coordinates + GT SDF values (for L_sdf + L_eik)
  - Unsupervised points: 3D coordinates only (for L_eik only)

The supervision ratio controls how many supervised points are used.
Points are pre-split by ratio during preprocessing.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SDFDataset(Dataset):
    """Dataset for loading preprocessed SDF samples.

    Each item corresponds to one shape and contains supervised + unsupervised
    point samples loaded from .npz files.

    Args:
        data_dir: base directory containing ratio_X/ subdirectories
        supervision_ratio: which ratio split to load (1.0, 0.5, 0.1, 0.05)
        split: "train" or "val" — uses first train_frac shapes for train
        train_frac: fraction of shapes used for training
        num_shapes: max number of shapes to use (-1 = all)
    """

    def __init__(
        self,
        data_dir: str,
        supervision_ratio: float = 1.0,
        split: str = "train",
        train_frac: float = 0.75,
        num_shapes: int = -1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.supervision_ratio = supervision_ratio
        self.split = split

        # Find the ratio directory
        ratio_str = f"ratio_{supervision_ratio:.2f}".replace(".", "p")
        self.ratio_dir = os.path.join(data_dir, ratio_str)

        if not os.path.isdir(self.ratio_dir):
            raise FileNotFoundError(
                f"Ratio directory not found: {self.ratio_dir}. "
                f"Run preprocess.py first."
            )

        # Discover all .npz files
        all_files = sorted(glob.glob(os.path.join(self.ratio_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No .npz files found in {self.ratio_dir}")

        # Globally shuffle before split (not stratified — category distribution may vary)
        rng = random.Random(seed)
        rng.shuffle(all_files)

        if num_shapes > 0:
            all_files = all_files[:num_shapes]

        # Train/val split (deterministic given seed, category-balanced)
        n_train = int(len(all_files) * train_frac)
        if split == "train":
            self.files = all_files[:n_train]
        else:
            self.files = all_files[n_train:]

        if not self.files:
            raise ValueError(
                f"No files for split '{split}': "
                f"{len(all_files)} total, {n_train} train"
            )

        # Extract shape names for identification
        self.shape_names = [
            os.path.splitext(os.path.basename(f))[0] for f in self.files
        ]

        # Preload all data into memory for fast training
        self.data = []
        for f in self.files:
            npz = np.load(f)
            self.data.append({
                "points_sup": torch.from_numpy(npz["points_sup"]),     # (N_sup, 3)
                "sdf_sup": torch.from_numpy(npz["sdf_sup"]),           # (N_sup,)
                "points_unsup": torch.from_numpy(npz["points_unsup"]), # (N_unsup, 3)
            })

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        """Return all data for a single shape.

        Returns dict with:
            points_sup: (N_sup, 3) supervised point coordinates
            sdf_sup: (N_sup,) ground truth SDF values
            points_unsup: (N_unsup, 3) unsupervised point coordinates
            shape_idx: int, index for latent code lookup
        """
        item = self.data[idx]
        return {
            "points_sup": item["points_sup"],
            "sdf_sup": item["sdf_sup"],
            "points_unsup": item["points_unsup"],
            "shape_idx": idx,
        }


def sample_batch_points(
    data: dict,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Sample a fixed-size batch of points from a shape's data.

    Splits batch_size between supervised and unsupervised points (50/50).
    Both sets are used for Eikonal; only supervised points contribute to L_sdf.

    Args:
        data: dict from SDFDataset.__getitem__
        batch_size: total number of points to sample
        device: torch device

    Returns:
        dict with:
            sup_points: (N_sup_batch, 3) with requires_grad=True
            sup_sdf: (N_sup_batch,)
            unsup_points: (N_unsup_batch, 3) with requires_grad=True
            shape_idx: int
    """
    n_sup_batch = batch_size // 2
    n_unsup_batch = batch_size - n_sup_batch

    # Sample supervised points
    sup_points = data["points_sup"]
    sup_sdf = data["sdf_sup"]
    n_sup_total = len(sup_points)
    sup_indices = torch.randint(0, n_sup_total, (n_sup_batch,))
    batch_sup_points = sup_points[sup_indices].to(device).requires_grad_(True)
    batch_sup_sdf = sup_sdf[sup_indices].to(device)

    # Sample unsupervised points
    unsup_points = data["points_unsup"]
    n_unsup_total = len(unsup_points)
    unsup_indices = torch.randint(0, n_unsup_total, (n_unsup_batch,))
    batch_unsup_points = unsup_points[unsup_indices].to(device).requires_grad_(True)

    return {
        "sup_points": batch_sup_points,
        "sup_sdf": batch_sup_sdf,
        "unsup_points": batch_unsup_points,
        "shape_idx": data["shape_idx"],
    }
