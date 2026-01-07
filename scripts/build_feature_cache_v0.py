#!/usr/bin/env python3
"""
Build feature cache from IQ dataset.

Converts IQ signals to features using the V0.3 extractor.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.features import extract_features, load_config as load_features_config
from common.data.io import get_class_mapping


def build_feature_cache(
    dataset_root: Path,
    features_config_path: Path,
    out_root: Path,
    verbose: bool = True
) -> dict:
    """
    Build feature cache from IQ dataset.
    
    Args:
        dataset_root: Path to dataset with IQ NPZ files.
        features_config_path: Path to features config YAML.
        out_root: Output directory for cached features.
        verbose: Print progress.
        
    Returns:
        Summary dict.
    """
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    features_config = load_features_config(features_config_path)
    
    splits = ["train", "val", "test_id", "test_ood_mod", "test_ood_chan"]
    summary = {"splits": {}}
    
    # Build class mapping (ID classes first, then OOD)
    id_classes = ["BPSK", "QPSK", "QAM16", "GFSK", "NOISE"]
    ood_classes = ["PSK8", "QAM64", "CPFSK"]
    class_to_idx = get_class_mapping(id_classes, ood_classes)
    
    for split in splits:
        iq_path = dataset_root / f"{split}.npz"
        if not iq_path.exists():
            if verbose:
                print(f"Skipping {split}: {iq_path} not found")
            continue
        
        if verbose:
            print(f"Processing {split}...")
        
        # Load IQ data
        data = np.load(iq_path, allow_pickle=True)
        iq_data = data["iq"]  # (N, 2048) complex64
        labels = data["y"]     # (N,) int32
        label_names = data["label_names"]  # (N,) str
        snr_db = data.get("snr_db", None)
        
        n_examples = len(iq_data)
        
        # Get feature shape from first example
        first_feat = extract_features(iq_data[0], features_config)
        feat_shape = first_feat.shape  # (1, 256, 15)
        
        # Allocate output arrays
        X = np.zeros((n_examples,) + feat_shape, dtype=np.float32)
        y = np.zeros(n_examples, dtype=np.int64)
        
        # Extract features for each example
        for i in tqdm(range(n_examples), desc=split, disable=not verbose):
            X[i] = extract_features(iq_data[i], features_config)
            y[i] = labels[i]
        
        # Save cached features
        out_path = out_root / f"{split}.npz"
        save_dict = {
            "X": X,
            "y": y,
            "label_names": label_names,
        }
        if snr_db is not None:
            save_dict["snr_db"] = snr_db
        
        np.savez_compressed(out_path, **save_dict)
        
        if verbose:
            print(f"  Saved {n_examples} examples to {out_path}")
            print(f"  X shape: {X.shape}, y shape: {y.shape}")
        
        summary["splits"][split] = {
            "n_examples": n_examples,
            "X_shape": list(X.shape),
            "path": str(out_path)
        }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build feature cache from IQ dataset")
    parser.add_argument(
        "--dataset_root", type=Path,
        default=project_root / "data" / "datasets" / "v0",
        help="Path to IQ dataset root"
    )
    parser.add_argument(
        "--features_config", type=Path,
        default=project_root / "common" / "features" / "configs" / "features_v0.yaml",
        help="Path to features config"
    )
    parser.add_argument(
        "--out_root", type=Path,
        default=project_root / "data" / "features" / "v0",
        help="Output directory for cached features"
    )
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Feature Cache Builder (V0)")
    print("=" * 60)
    print()
    
    summary = build_feature_cache(
        dataset_root=args.dataset_root,
        features_config_path=args.features_config,
        out_root=args.out_root,
        verbose=not args.quiet
    )
    
    print()
    print("Done!")
    print(f"Output: {args.out_root}")


if __name__ == "__main__":
    main()
