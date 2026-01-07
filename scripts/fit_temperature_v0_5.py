#!/usr/bin/env python3
"""
Fit temperature scaling on validation set.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tfm_ai_utamed.assurance.temperature_scaling import (
    TemperatureScaling, find_threshold_for_coverage
)


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in: {runs_dir}")
    return runs[-1]


def fit_temperature(
    run_dir: Path,
    coverage_target: float = 0.95,
    verbose: bool = True
) -> dict:
    """
    Fit temperature and threshold on validation set.
    
    Args:
        run_dir: V0.5 run directory with logits_val.npz.
        coverage_target: Target coverage for abstention threshold.
        verbose: Print progress.
        
    Returns:
        Summary dict.
    """
    run_dir = Path(run_dir)
    
    # Load validation logits
    val_path = run_dir / "logits_val.npz"
    data = np.load(val_path)
    logits = data["logits"]
    y = data["y"]
    
    if verbose:
        print(f"Loaded {len(y)} validation examples")
    
    # Fit temperature
    ts = TemperatureScaling()
    T = ts.fit(logits, y)
    
    if verbose:
        print(f"Fitted temperature T = {T:.4f}")
        print(f"  NLL before: {ts.nll_before:.4f}")
        print(f"  NLL after:  {ts.nll_after:.4f}")
    
    # Save temperature
    temp_path = run_dir / "temperature.json"
    with open(temp_path, "w") as f:
        json.dump(ts.to_dict(), f, indent=2)
    
    if verbose:
        print(f"Saved to: {temp_path}")
    
    # Get calibrated probs
    probs = ts.calibrate(logits)
    confidences = np.max(probs, axis=1)
    
    # Find threshold for target coverage
    threshold, actual_coverage = find_threshold_for_coverage(confidences, coverage_target)
    
    if verbose:
        print()
        print(f"Threshold for {coverage_target:.0%} coverage:")
        print(f"  τ = {threshold:.4f}")
        print(f"  Actual coverage: {actual_coverage:.4f}")
    
    # Save threshold
    thresh_path = run_dir / "threshold.json"
    thresh_data = {
        "threshold": threshold,
        "coverage_target": coverage_target,
        "coverage_actual": actual_coverage,
        "temperature": T,
    }
    with open(thresh_path, "w") as f:
        json.dump(thresh_data, f, indent=2)
    
    if verbose:
        print(f"Saved to: {thresh_path}")
    
    return {
        "temperature": T,
        "threshold": threshold,
        "coverage_target": coverage_target,
        "coverage_actual": actual_coverage,
    }


def main():
    parser = argparse.ArgumentParser(description="Fit temperature scaling")
    parser.add_argument(
        "--run_dir", type=Path, default=None,
        help="V0.5 run directory (default: latest)"
    )
    parser.add_argument(
        "--coverage", type=float, default=0.95,
        help="Target coverage for threshold"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V0.5 Temperature Scaling")
    print("=" * 60)
    print()
    
    if args.run_dir is None:
        args.run_dir = find_latest_run(project_root / "runs" / "v0_5")
    
    print(f"Run directory: {args.run_dir}")
    print()
    
    fit_temperature(args.run_dir, args.coverage)
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
