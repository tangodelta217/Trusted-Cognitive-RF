#!/usr/bin/env python3
"""
Evaluation tool with calibration and assurance metrics.

Usage:
    python -m tools.eval                    # Basic evaluation
    python -m tools.eval --with-calibration # Include calibration analysis
    python -m tools.eval --help

Generates:
    - artifacts/policy/temperature.json
    - reports/metrics/assurance.json
    - reports/figures/reliability.png
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_latest_logits_run() -> Optional[Path]:
    """Find the latest run with logits files."""
    runs_v05 = project_root / "runs" / "v0_5"
    if not runs_v05.exists():
        return None
    
    runs = sorted([r for r in runs_v05.iterdir() if r.is_dir()])
    for run_dir in reversed(runs):
        if (run_dir / "logits_val.npz").exists():
            return run_dir
    return None


def load_logits(run_dir: Path, split: str) -> tuple:
    """Load logits and labels for a split."""
    path = run_dir / f"logits_{split}.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["logits"].astype(np.float32), data["y"].astype(np.int64)


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Fit temperature scaling on validation data."""
    from tfm_ai_utamed.assurance.temperature_scaling import TemperatureScaling
    
    ts = TemperatureScaling()
    T = ts.fit(logits, labels)
    
    return {
        "temperature": float(T),
        "nll_before": float(ts.nll_before),
        "nll_after": float(ts.nll_after),
        "fitted_on": "val",
        "n_samples": len(labels),
    }


def compute_ece_metrics(logits: np.ndarray, labels: np.ndarray, 
                        temperature: float = 1.0, n_bins: int = 15) -> Dict[str, Any]:
    """Compute ECE metrics before and after calibration."""
    from tfm_ai_utamed.assurance.temperature_scaling import softmax
    from tfm_ai_utamed.assurance.calibration_metrics import compute_ece, reliability_diagram_data
    
    probs_before = softmax(logits, 1.0)
    probs_after = softmax(logits, temperature)
    
    ece_before, bin_data_before = compute_ece(probs_before, labels, n_bins)
    ece_after, bin_data_after = compute_ece(probs_after, labels, n_bins)
    
    # Compute accuracy
    preds_before = np.argmax(probs_before, axis=1)
    preds_after = np.argmax(probs_after, axis=1)
    accuracy = float(np.mean(preds_after == labels))
    
    return {
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "ece_reduction": round(ece_before - ece_after, 6),
        "ece_reduction_pct": round((ece_before - ece_after) / ece_before * 100, 2) if ece_before > 0 else 0.0,
        "accuracy": round(accuracy, 4),
        "n_samples": len(labels),
        "n_bins": n_bins,
        "bin_data": {
            "centers": bin_data_after["bin_lowers"].tolist(),
            "accuracies": bin_data_after["bin_accuracies"].tolist(),
            "confidences": bin_data_after["bin_confidences"].tolist(),
            "counts": bin_data_after["bin_counts"].tolist(),
        }
    }


def generate_reliability_diagram(bin_data: Dict, title: str, save_path: Path) -> None:
    """Generate and save reliability diagram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    bin_centers = np.array(bin_data["centers"]) + 1/(2*len(bin_data["centers"]))
    bin_accuracies = np.array(bin_data["accuracies"])
    bin_counts = np.array(bin_data["counts"])
    
    # Only plot bins with samples
    mask = bin_counts > 0
    width = 0.9 / len(bin_centers)
    
    # Bar chart
    bars = ax.bar(bin_centers[mask], bin_accuracies[mask], width=width,
                  alpha=0.7, color='steelblue', edgecolor='navy', label='Accuracy')
    
    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    
    # Gap visualization
    for i, (bc, ba) in enumerate(zip(bin_centers[mask], bin_accuracies[mask])):
        if abs(bc - ba) > 0.02:  # Only show significant gaps
            ax.plot([bc, bc], [min(bc, ba), max(bc, ba)], 
                    'orange', linewidth=2, alpha=0.7)
    
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add sample count annotation
    total_samples = sum(bin_counts)
    ax.text(0.95, 0.05, f"n={total_samples}", transform=ax.transAxes,
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_ece_table(metrics: Dict[str, Dict]) -> None:
    """Print ECE comparison table."""
    print()
    print("=" * 70)
    print("ECE Comparison (Expected Calibration Error)")
    print("=" * 70)
    print(f"{'Split':<15} {'ECE Before':<12} {'ECE After':<12} {'Reduction':<12} {'Accuracy':<10}")
    print("-" * 70)
    
    for split, data in metrics.items():
        if data is None:
            continue
        print(f"{split:<15} {data['ece_before']:<12.4f} {data['ece_after']:<12.4f} "
              f"{data['ece_reduction_pct']:>8.1f}%    {data['accuracy']:<10.4f}")
    
    print("=" * 70)


def run_evaluation(run_dir: Path, with_calibration: bool = False, 
                   verbose: bool = True) -> Dict[str, Any]:
    """Run full evaluation with optional calibration analysis."""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "with_calibration": with_calibration,
    }
    
    # Load validation logits
    val_logits, val_labels = load_logits(run_dir, "val")
    if val_logits is None:
        print("ERROR: No validation logits found. Run collect_logits first.")
        return results
    
    if verbose:
        print(f"Loaded validation logits: {val_logits.shape}")
    
    # Fit temperature scaling
    if verbose:
        print("\nFitting temperature scaling on validation set...")
    
    temp_result = fit_temperature_scaling(val_logits, val_labels)
    T = temp_result["temperature"]
    
    if verbose:
        print(f"  Optimal temperature: T = {T:.4f}")
        print(f"  NLL: {temp_result['nll_before']:.4f} → {temp_result['nll_after']:.4f}")
    
    results["temperature_scaling"] = temp_result
    
    # Save temperature to artifacts
    artifacts_dir = project_root / "artifacts" / "policy"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    temp_path = artifacts_dir / "temperature.json"
    with open(temp_path, "w") as f:
        json.dump(temp_result, f, indent=2)
    
    if verbose:
        print(f"  Saved: {temp_path}")
    
    # Compute ECE metrics for all splits
    splits = ["val", "test_id", "test_ood_chan"]
    ece_metrics = {}
    
    if verbose:
        print("\nComputing ECE metrics...")
    
    for split in splits:
        logits, labels = load_logits(run_dir, split)
        if logits is None:
            ece_metrics[split] = None
            continue
        
        metrics = compute_ece_metrics(logits, labels, T)
        ece_metrics[split] = metrics
        
        if verbose:
            print(f"  {split}: ECE {metrics['ece_before']:.4f} → {metrics['ece_after']:.4f} "
                  f"({metrics['ece_reduction_pct']:.1f}% reduction)")
    
    results["ece_metrics"] = ece_metrics
    
    # Print ECE table
    if with_calibration and verbose:
        print_ece_table(ece_metrics)
    
    # Generate reliability diagram
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if ece_metrics.get("val") is not None:
        reliability_path = figures_dir / "reliability.png"
        generate_reliability_diagram(
            ece_metrics["val"]["bin_data"],
            f"Reliability Diagram (val, T={T:.2f})",
            reliability_path
        )
        results["reliability_diagram"] = str(reliability_path)
        
        if verbose:
            print(f"\nReliability diagram saved: {reliability_path}")
    
    # Save assurance metrics
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    assurance_path = metrics_dir / "assurance.json"
    
    with open(assurance_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    if verbose:
        print(f"Assurance metrics saved: {assurance_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation tool with calibration and assurance metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m tools.eval                     # Basic evaluation
    python -m tools.eval --with-calibration  # Full calibration analysis
    python -m tools.eval --run-dir runs/v0_5/run_XYZ
        """
    )
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run directory with logits (default: latest v0_5 run)")
    parser.add_argument("--with-calibration", action="store_true",
                        help="Include calibration analysis and ECE table")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Evaluation Tool — Cognitive Trusted RF Receiver")
    print("=" * 60)
    
    # Find run directory
    if args.run_dir is None:
        args.run_dir = find_latest_logits_run()
        if args.run_dir is None:
            print("ERROR: No runs found with logits. Run the training pipeline first.")
            sys.exit(1)
    
    print(f"Run directory: {args.run_dir}")
    
    # Run evaluation
    results = run_evaluation(
        args.run_dir, 
        with_calibration=args.with_calibration,
        verbose=not args.quiet
    )
    
    # Check ECE improvement
    val_metrics = results.get("ece_metrics", {}).get("val")
    if val_metrics:
        improvement = val_metrics["ece_reduction_pct"]
        if improvement >= 20:
            print(f"\n✓ ECE improvement: {improvement:.1f}% (threshold: 20%)")
        else:
            print(f"\n⚠ ECE improvement: {improvement:.1f}% (below 20% threshold)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
