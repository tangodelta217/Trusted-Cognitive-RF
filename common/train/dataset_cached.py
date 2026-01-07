"""
PyTorch Dataset for cached features.

Loads pre-extracted features from NPZ files for fast training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Any


class CachedFeaturesDataset(Dataset):
    """
    Dataset that loads cached features from NPZ files.
    
    Expected NPZ format:
        - X: float32 (N, C, F, T) features
        - y: int64 (N,) labels
    """
    
    def __init__(
        self,
        features_path: Path,
        transform: Optional[callable] = None
    ):
        """
        Args:
            features_path: Path to cached features NPZ file.
            transform: Optional transform to apply to features.
        """
        self.features_path = Path(features_path)
        self.transform = transform
        
        # Load data
        data = np.load(self.features_path)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.int64)
        
        # Optional metadata
        self.snr_db = data.get("snr_db", None)
        self.label_names = data.get("label_names", None)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> tuple:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
    
    @property
    def num_classes(self) -> int:
        return len(np.unique(self.y))
    
    @property
    def class_counts(self) -> Dict[int, int]:
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


def load_split_datasets(
    features_root: Path,
    splits: list = None
) -> Dict[str, CachedFeaturesDataset]:
    """
    Load multiple split datasets from a features cache directory.
    
    Args:
        features_root: Root directory containing {split}.npz files.
        splits: List of split names to load. Default: all standard splits.
        
    Returns:
        Dict mapping split name to dataset.
    """
    features_root = Path(features_root)
    
    if splits is None:
        splits = ["train", "val", "test_id", "test_ood_mod", "test_ood_chan"]
    
    datasets = {}
    for split in splits:
        path = features_root / f"{split}.npz"
        if path.exists():
            datasets[split] = CachedFeaturesDataset(path)
    
    return datasets
