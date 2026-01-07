#!/usr/bin/env python3
"""
Sweep operating points for coverage vs accuracy vs OOD rejection.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
import csv
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.policy.abstention import (
    sweep_operating_points, PRESETS, fit_threshold_by_coverage
)
from tfm_ai_utamed.assurance.temperature_scaling import softmax

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in: {runs_dir}")
    return runs[-1]


def run_sweep(
    run_v0_5: Path,
    out_dir: Path,
    verbose: bool = True
) -> dict:
    """
    Run operating point sweep.
    """
    run_v0_5 = Path(run_v0_5)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load temperature
    with open(run_v0_5 / "temperature.json", "r") as f:
        temp_data = json.load(f)
    T = temp_data["temperature"]
    
    if verbose:
        print(f"Temperature: {T:.4f}")
    
    # Load logits
    def load_split(name):
        path = run_v0_5 / f"logits_{name}.npz"
        if not path.exists():
            return None, None, None
        data = np.load(path)
        logits = data["logits"]
        y = data["y"]
        probs = softmax(logits, T)
        conf = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        return conf, y, preds
    
    conf_val, y_val, preds_val = load_split("val")
    conf_test_id, y_test_id, preds_test_id = load_split("test_id")
    conf_ood_chan, y_ood_chan, preds_ood_chan = load_split("test_ood_chan")
    conf_ood_mod, _, _ = load_split("test_ood_mod")
    
    if conf_val is None:
        raise ValueError("No validation logits found")
    
    # Coverage targets to sweep
    coverage_targets = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    
    # Sweep for test_id
    results_id = sweep_operating_points(
        conf_val, conf_test_id, y_test_id, preds_test_id, conf_ood_mod,
        coverage_targets
    )
    
    # Sweep for test_ood_chan
    results_chan = sweep_operating_points(
        conf_val, conf_ood_chan, y_ood_chan, preds_ood_chan, conf_ood_mod,
        coverage_targets
    )
    
    # Write sweep.csv
    csv_path = out_dir / "sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "coverage_target", "tau", 
            "coverage_test_id", "acc_accepted_test_id",
            "coverage_ood_chan", "acc_accepted_ood_chan",
            "ood_mod_rejection"
        ])
        writer.writeheader()
        
        for r_id, r_chan in zip(results_id, results_chan):
            writer.writerow({
                "coverage_target": r_id["coverage_target"],
                "tau": f"{r_id['tau']:.4f}",
                "coverage_test_id": f"{r_id['coverage_actual']:.4f}",
                "acc_accepted_test_id": f"{r_id['accuracy_accepted']:.4f}",
                "coverage_ood_chan": f"{r_chan['coverage_actual']:.4f}",
                "acc_accepted_ood_chan": f"{r_chan['accuracy_accepted']:.4f}",
                "ood_mod_rejection": f"{r_id['ood_rejection']:.4f}",
            })
    
    if verbose:
        print(f"Saved: {csv_path}")
    
    # Generate presets
    presets_data = {}
    for name, preset in PRESETS.items():
        tau = fit_threshold_by_coverage(conf_val, preset["coverage_target"])
        presets_data[name] = {
            "coverage_target": preset["coverage_target"],
            "tau": float(tau),
            "description": preset["description"],
        }
    
    # Save summary
    summary = {
        "temperature": T,
        "presets": presets_data,
        "sweep_coverage_targets": coverage_targets,
    }
    with open(out_dir / "operating_points.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"Saved: {out_dir / 'operating_points.json'}")
    
    # === Plots ===
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: Coverage vs Accuracy Accepted
    fig, ax = plt.subplots(figsize=(8, 5))
    coverages_id = [r["coverage_actual"] for r in results_id]
    acc_id = [r["accuracy_accepted"] for r in results_id]
    coverages_chan = [r["coverage_actual"] for r in results_chan]
    acc_chan = [r["accuracy_accepted"] for r in results_chan]
    
    ax.plot(coverages_id, acc_id, 'o-', label="test_id", linewidth=2, markersize=6)
    ax.plot(coverages_chan, acc_chan, 's-', label="test_ood_chan", linewidth=2, markersize=6)
    
    # Mark presets
    for name, p in presets_data.items():
        tau = p["tau"]
        cov = np.mean(conf_test_id >= tau)
        acc = np.mean(preds_test_id[conf_test_id >= tau] == y_test_id[conf_test_id >= tau]) if cov > 0 else 0
        ax.axvline(cov, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(name, (cov, acc), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy (Accepted)")
    ax.set_title("Coverage vs Accuracy Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(plots_dir / "coverage_vs_accuracy.png", dpi=150)
    plt.close()
    
    if verbose:
        print(f"Saved: {plots_dir / 'coverage_vs_accuracy.png'}")
    
    # Plot 2: Coverage vs OOD Rejection
    fig, ax = plt.subplots(figsize=(8, 5))
    coverages = [r["coverage_actual"] for r in results_id]
    ood_rej = [r["ood_rejection"] for r in results_id]
    
    ax.plot(coverages, ood_rej, 'o-', color='coral', linewidth=2, markersize=6)
    ax.fill_between(coverages, ood_rej, alpha=0.3, color='coral')
    
    ax.set_xlabel("Coverage (on test_id)")
    ax.set_ylabel("OOD-MOD Rejection Rate")
    ax.set_title("Coverage vs OOD Rejection")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(plots_dir / "coverage_vs_ood_rejection.png", dpi=150)
    plt.close()
    
    if verbose:
        print(f"Saved: {plots_dir / 'coverage_vs_ood_rejection.png'}")
    
    # Plot 3: Combined trade-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(coverages_id, acc_id, 'o-', linewidth=2, label="Acc (test_id)")
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Accuracy Accepted")
    ax1.set_title("More Selective → Higher Quality")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(coverages, ood_rej, 'o-', color='coral', linewidth=2)
    ax2.set_xlabel("Coverage")
    ax2.set_ylabel("OOD Rejection Rate")
    ax2.set_title("More Selective → More OOD Rejected")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "tradeoff_summary.png", dpi=150)
    plt.close()
    
    if verbose:
        print(f"Saved: {plots_dir / 'tradeoff_summary.png'}")
        print()
        print("=== Preset Operating Points ===")
        for name, p in presets_data.items():
            print(f"  {name}: τ={p['tau']:.4f} (coverage={p['coverage_target']})")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Sweep operating points")
    parser.add_argument("--run_v0_5", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V0.6 Operating Point Sweep")
    print("=" * 60)
    print()
    
    if args.run_v0_5 is None:
        args.run_v0_5 = find_latest_run(project_root / "runs" / "v0_5")
    
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = project_root / "runs" / "v0_6" / f"run_{timestamp}"
    
    print(f"V0.5 run: {args.run_v0_5}")
    print(f"Output: {args.out_dir}")
    print()
    
    run_sweep(args.run_v0_5, args.out_dir)
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
