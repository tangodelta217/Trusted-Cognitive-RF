#!/usr/bin/env python3
"""
Collect logits from trained model for all splits.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.infer.predict import load_model, predict_logits
from common.train.dataset_cached import CachedFeaturesDataset
from datetime import datetime


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in: {runs_dir}")
    return runs[-1]


def collect_logits(
    run_v0_4: Path,
    features_root: Path,
    out_root: Path,
    model_config: Path,
    verbose: bool = True
) -> dict:
    """
    Collect logits for all splits.
    
    Args:
        run_v0_4: Path to V0.4 run directory.
        features_root: Path to cached features.
        out_root: Output directory for V0.5.
        model_config: Path to model config.
        verbose: Print progress.
        
    Returns:
        Summary dict.
    """
    run_v0_4 = Path(run_v0_4)
    features_root = Path(features_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Load model
    checkpoint = run_v0_4 / "best_model.pt"
    model = load_model(checkpoint, model_config)
    
    if verbose:
        print(f"Loaded model from: {checkpoint}")
    
    splits = ["val", "test_id", "test_ood_mod", "test_ood_chan"]
    summary = {"splits": {}}
    
    for split in splits:
        features_path = features_root / f"{split}.npz"
        if not features_path.exists():
            if verbose:
                print(f"Skipping {split}: not found")
            continue
        
        if verbose:
            print(f"Processing {split}...")
        
        # Load features
        dataset = CachedFeaturesDataset(features_path)
        X = dataset.X
        y = dataset.y
        
        # Get logits
        logits = predict_logits(model, X)
        
        # Save
        out_path = out_root / f"logits_{split}.npz"
        np.savez_compressed(
            out_path,
            logits=logits,
            y=y,
            label_names=dataset.label_names if hasattr(dataset, 'label_names') else None
        )
        
        if verbose:
            print(f"  Saved {len(y)} examples to {out_path}")
        
        summary["splits"][split] = {
            "n_examples": len(y),
            "logits_shape": list(logits.shape),
            "path": str(out_path)
        }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect logits from V0.4 model")
    parser.add_argument(
        "--run_v0_4", type=Path, default=None,
        help="V0.4 run directory (default: latest)"
    )
    parser.add_argument(
        "--features_root", type=Path,
        default=project_root / "data" / "features" / "v0"
    )
    parser.add_argument(
        "--out_root", type=Path, default=None,
        help="Output directory for V0.5 (default: runs/v0_5/run_<timestamp>)"
    )
    parser.add_argument(
        "--model_config", type=Path,
        default=project_root / "common" / "models" / "configs" / "model_cnn_small_v0.yaml"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V0.5 Logits Collection")
    print("=" * 60)
    print()
    
    # Find V0.4 run
    if args.run_v0_4 is None:
        args.run_v0_4 = find_latest_run(project_root / "runs" / "v0_4")
    
    # Create V0.5 run dir
    if args.out_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_root = project_root / "runs" / "v0_5" / f"run_{timestamp}"
    
    print(f"V0.4 run: {args.run_v0_4}")
    print(f"Features: {args.features_root}")
    print(f"Output: {args.out_root}")
    print()
    
    summary = collect_logits(
        run_v0_4=args.run_v0_4,
        features_root=args.features_root,
        out_root=args.out_root,
        model_config=args.model_config
    )
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
