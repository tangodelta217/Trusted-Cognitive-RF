#!/usr/bin/env python3
"""
CLI wrapper for evaluating V0.4 model.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.eval.eval_v0_4 import evaluate_model


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in: {runs_dir}")
    
    return runs[-1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate V0.4 model")
    parser.add_argument(
        "--run_dir", type=Path, default=None,
        help="Run directory (default: latest in runs/v0_4)"
    )
    parser.add_argument(
        "--features_root", type=Path,
        default=project_root / "data" / "features" / "v0",
        help="Path to cached features"
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=["test_id", "test_ood_mod", "test_ood_chan"],
        help="Splits to evaluate"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V0.4 Model Evaluation")
    print("=" * 60)
    print()
    
    # Find run directory
    if args.run_dir is None:
        args.run_dir = find_latest_run(project_root / "runs" / "v0_4")
    
    print(f"Run directory: {args.run_dir}")
    print(f"Features root: {args.features_root}")
    print(f"Splits: {args.splits}")
    print()
    
    report = evaluate_model(
        run_dir=args.run_dir,
        features_root=args.features_root,
        splits=args.splits,
        verbose=True
    )
    
    print()
    print("=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    
    for split, metrics in report["splits"].items():
        print(f"{split}: Accuracy={metrics['accuracy']:.4f}, Macro F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
