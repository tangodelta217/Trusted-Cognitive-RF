#!/usr/bin/env python3
"""
OOD Detection Evaluation — Entropy and Energy-based methods.

Usage:
    python -m tools.ood_eval
    python -m tools.ood_eval --id_split test_id --ood_split test_ood_mod
    python -m tools.ood_eval --methods entropy,energy

Computes AUROC and AUPR for OOD detection and generates ROC curves.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_latest_logits_run() -> Path:
    """Find the latest run with logits files."""
    runs_v05 = project_root / "runs" / "v0_5"
    if not runs_v05.exists():
        return None
    
    runs = sorted([r for r in runs_v05.iterdir() if r.is_dir()])
    for run_dir in reversed(runs):
        if (run_dir / "logits_val.npz").exists():
            return run_dir
    return None


def load_logits(run_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load logits and labels for a split."""
    path = run_dir / f"logits_{split}.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["logits"].astype(np.float32), data["y"].astype(np.int64)


def load_temperature() -> float:
    """Load calibrated temperature."""
    temp_path = project_root / "artifacts" / "policy" / "temperature.json"
    if temp_path.exists():
        with open(temp_path) as f:
            return json.load(f)["temperature"]
    
    # Try runs/v0_5
    runs_v05 = project_root / "runs" / "v0_5"
    if runs_v05.exists():
        runs = sorted([r for r in runs_v05.iterdir() if r.is_dir()])
        if runs:
            temp_path = runs[-1] / "temperature.json"
            if temp_path.exists():
                with open(temp_path) as f:
                    return json.load(f)["temperature"]
    
    return 1.0


def compute_ood_scores(logits_id: np.ndarray, logits_ood: np.ndarray, 
                        temperature: float) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute OOD scores for ID and OOD data."""
    from tfm_ai_utamed.assurance.temperature_scaling import softmax
    from tfm_ai_utamed.assurance.ood_scores import compute_all_scores
    
    probs_id = softmax(logits_id, temperature)
    probs_ood = softmax(logits_ood, temperature)
    
    scores_id = compute_all_scores(logits_id, probs_id, temperature)
    scores_ood = compute_all_scores(logits_ood, probs_ood, temperature)
    
    return {
        "id": scores_id,
        "ood": scores_ood,
    }


def evaluate_ood_method(scores_id: np.ndarray, scores_ood: np.ndarray, 
                         method_name: str) -> Dict[str, float]:
    """Evaluate a single OOD detection method."""
    from tfm_ai_utamed.assurance.ood_scores import compute_auroc, compute_aupr
    
    auroc = compute_auroc(scores_id, scores_ood)
    aupr = compute_aupr(scores_id, scores_ood)
    
    return {
        "method": method_name,
        "auroc": round(auroc, 4),
        "aupr": round(aupr, 4),
    }


def plot_roc_curves(scores: Dict, methods: List[str], save_path: Path) -> None:
    """Generate ROC curves for OOD detection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tfm_ai_utamed.assurance.plots import compute_roc_curve
    from tfm_ai_utamed.assurance.ood_scores import compute_auroc
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = {'entropy': 'blue', 'energy': 'green', 'neg_msp': 'orange'}
    
    for method in methods:
        if method not in scores["id"]:
            continue
        
        roc_data = compute_roc_curve(scores["id"][method], scores["ood"][method])
        auroc = compute_auroc(scores["id"][method], scores["ood"][method])
        
        ax.plot(roc_data["fpr"], roc_data["tpr"], 
                color=colors.get(method, 'gray'),
                linewidth=2, 
                label=f'{method} (AUROC={auroc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel("False Positive Rate (ID classified as OOD)", fontsize=12)
    ax.set_ylabel("True Positive Rate (OOD detected)", fontsize=12)
    ax.set_title("OOD Detection ROC Curves", fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_score_distributions(scores: Dict, method: str, save_path: Path) -> None:
    """Plot score distributions for ID and OOD."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = 50
    ax.hist(scores["id"][method], bins=bins, alpha=0.6, 
            label="ID", color="steelblue", density=True)
    ax.hist(scores["ood"][method], bins=bins, alpha=0.6, 
            label="OOD", color="coral", density=True)
    
    ax.set_xlabel(f"{method.capitalize()} Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Score Distribution: {method}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="OOD Detection Evaluation"
    )
    parser.add_argument("--id_split", type=str, default="test_id",
                        help="ID split name")
    parser.add_argument("--ood_split", type=str, default="test_ood_mod",
                        help="OOD split name")
    parser.add_argument("--methods", type=str, default="entropy,energy",
                        help="Comma-separated list of methods")
    parser.add_argument("--run_dir", type=Path, default=None,
                        help="Run directory with logits")
    
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    
    print("=" * 60)
    print("OOD Detection Evaluation")
    print("=" * 60)
    
    # Find run directory
    if args.run_dir is None:
        args.run_dir = find_latest_logits_run()
        if args.run_dir is None:
            print("ERROR: No runs found with logits.")
            sys.exit(1)
    
    print(f"Run directory: {args.run_dir}")
    print(f"ID split: {args.id_split}")
    print(f"OOD split: {args.ood_split}")
    print(f"Methods: {methods}")
    print()
    
    # Load data
    logits_id, labels_id = load_logits(args.run_dir, args.id_split)
    logits_ood, labels_ood = load_logits(args.run_dir, args.ood_split)
    
    if logits_id is None:
        print(f"ERROR: Could not load ID split: {args.id_split}")
        sys.exit(1)
    
    if logits_ood is None:
        print(f"ERROR: Could not load OOD split: {args.ood_split}")
        sys.exit(1)
    
    print(f"ID samples: {len(labels_id)}")
    print(f"OOD samples: {len(labels_ood)}")
    
    # Load temperature
    T = load_temperature()
    print(f"Temperature: {T:.4f}")
    print()
    
    # Compute scores
    scores = compute_ood_scores(logits_id, logits_ood, T)
    
    # Evaluate methods
    results = {
        "timestamp": datetime.now().isoformat(),
        "id_split": args.id_split,
        "ood_split": args.ood_split,
        "temperature": T,
        "n_id": len(labels_id),
        "n_ood": len(labels_ood),
        "methods": {},
    }
    
    print("Results:")
    print("-" * 40)
    
    best_auroc = 0
    best_method = None
    
    for method in methods:
        # Map method name to score key
        score_key = method
        if method == "msp":
            score_key = "neg_msp"  # Use negative MSP (higher = more OOD)
        
        if score_key not in scores["id"]:
            print(f"  {method}: SKIP (score not available)")
            continue
        
        result = evaluate_ood_method(
            scores["id"][score_key], 
            scores["ood"][score_key],
            method
        )
        results["methods"][method] = result
        
        print(f"  {method}:")
        print(f"    AUROC: {result['auroc']:.4f}")
        print(f"    AUPR:  {result['aupr']:.4f}")
        
        if result["auroc"] > best_auroc:
            best_auroc = result["auroc"]
            best_method = method
    
    results["best_method"] = best_method
    results["best_auroc"] = best_auroc
    
    print()
    print(f"Best method: {best_method} (AUROC={best_auroc:.4f})")
    
    # Check threshold
    if best_auroc >= 0.65:
        print(f"✓ AUROC >= 0.65 threshold met!")
    else:
        print(f"⚠ AUROC {best_auroc:.4f} < 0.65 threshold")
        print("  Suggestions:")
        print("  - Increase OOD separation in synthetic generator")
        print("  - Try different OOD split (e.g., test_ood_chan)")
        print("  - Switch method (energy often better than entropy)")
    
    # Save results
    reports_dir = project_root / "reports"
    metrics_dir = reports_dir / "metrics"
    figures_dir = reports_dir / "figures"
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    ood_json_path = metrics_dir / "ood.json"
    with open(ood_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {ood_json_path}")
    
    # Generate ROC curves
    roc_path = figures_dir / "ood_roc.png"
    plot_roc_curves(scores, methods, roc_path)
    print(f"Saved: {roc_path}")
    
    # Generate score distribution for best method
    if best_method:
        score_key = best_method if best_method != "msp" else "neg_msp"
        dist_path = figures_dir / f"ood_distribution_{best_method}.png"
        plot_score_distributions(scores, score_key, dist_path)
        print(f"Saved: {dist_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
