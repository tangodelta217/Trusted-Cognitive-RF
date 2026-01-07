#!/usr/bin/env python3
"""
Generate assurance plots: reliability diagrams, risk-coverage, OOD ROC.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tfm_ai_utamed.assurance.plots import (
    plot_reliability_diagram,
    plot_risk_coverage,
    plot_ood_roc,
    plot_confidence_histograms,
    compute_roc_curve,
)


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in: {runs_dir}")
    return runs[-1]


def plot_assurance(run_dir: Path, verbose: bool = True) -> None:
    """Generate all assurance plots."""
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Load report for threshold
    with open(run_dir / "assurance_report.json", "r") as f:
        report = json.load(f)
    threshold = report["threshold"]
    
    # === Reliability diagrams ===
    for split in ["val", "test_id"]:
        rel_path = run_dir / f"reliability_{split}.npz"
        if rel_path.exists():
            data = dict(np.load(rel_path))
            plot_reliability_diagram(
                data,
                title=f"Reliability Diagram ({split})",
                save_path=plots_dir / f"reliability_{split}.png"
            )
            if verbose:
                print(f"Saved: reliability_{split}.png")
    
    # === Risk-coverage ===
    for split in ["test_id", "test_ood_chan"]:
        rc_path = run_dir / f"risk_coverage_{split}.npz"
        if rc_path.exists():
            data = np.load(rc_path)
            plot_risk_coverage(
                data["coverages"],
                data["risks"],
                title=f"Risk-Coverage ({split})",
                save_path=plots_dir / f"risk_coverage_{split}.png"
            )
            if verbose:
                print(f"Saved: risk_coverage_{split}.png")
    
    # === OOD detection ===
    ood_path = run_dir / "ood_scores.npz"
    if ood_path.exists():
        data = np.load(ood_path)
        conf_id = data["conf_id"]
        conf_ood = data["conf_ood"]
        
        # Use negative MSP (higher = more OOD)
        neg_msp_id = -conf_id
        neg_msp_ood = -conf_ood
        
        # ROC curve
        roc = compute_roc_curve(neg_msp_id, neg_msp_ood)
        
        # Get AUROC from report
        auroc = report["ood_detection"]["metrics"]["neg_msp"]["auroc"]
        
        plot_ood_roc(
            roc["fpr"],
            roc["tpr"],
            auroc,
            title="OOD Detection ROC (MSP)",
            save_path=plots_dir / "ood_roc.png"
        )
        if verbose:
            print(f"Saved: ood_roc.png")
        
        # Confidence histograms
        plot_confidence_histograms(
            conf_id,
            conf_ood,
            threshold=threshold,
            title="Confidence Distribution (ID vs OOD)",
            save_path=plots_dir / "confidence_histogram.png"
        )
        if verbose:
            print(f"Saved: confidence_histogram.png")
    
    if verbose:
        print()
        print(f"All plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate assurance plots")
    parser.add_argument(
        "--run_dir", type=Path, default=None,
        help="V0.5 run directory"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V0.5 Assurance Plots")
    print("=" * 60)
    print()
    
    if args.run_dir is None:
        args.run_dir = find_latest_run(project_root / "runs" / "v0_5")
    
    print(f"Run directory: {args.run_dir}")
    print()
    
    plot_assurance(args.run_dir)
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
