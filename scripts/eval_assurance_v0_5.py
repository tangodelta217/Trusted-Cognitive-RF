#!/usr/bin/env python3
"""
Evaluate assurance metrics: ECE, risk-coverage, OOD detection.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tfm_ai_utamed.assurance.temperature_scaling import TemperatureScaling, softmax
from tfm_ai_utamed.assurance.calibration_metrics import compute_ece, reliability_diagram_data
from tfm_ai_utamed.assurance.ood_scores import (
    compute_all_scores, compute_auroc, compute_aupr
)
from tfm_ai_utamed.assurance.risk_coverage import (
    compute_risk_coverage, abstention_stats, compute_auc_risk_coverage
)

# ID classes only
ID_CLASSES = ["BPSK", "QPSK", "QAM16", "GFSK", "NOISE"]
ID_CLASS_INDICES = list(range(5))


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in: {runs_dir}")
    return runs[-1]


def load_logits(run_dir: Path, split: str):
    """Load logits for a split."""
    path = run_dir / f"logits_{split}.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["logits"], data["y"]


def evaluate_assurance(run_dir: Path, verbose: bool = True) -> dict:
    """
    Evaluate all assurance metrics.
    """
    run_dir = Path(run_dir)
    
    # Load temperature
    temp_path = run_dir / "temperature.json"
    with open(temp_path, "r") as f:
        temp_data = json.load(f)
    T = temp_data["temperature"]
    
    # Load threshold
    thresh_path = run_dir / "threshold.json"
    with open(thresh_path, "r") as f:
        thresh_data = json.load(f)
    threshold = thresh_data["threshold"]
    
    if verbose:
        print(f"Temperature: {T:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print()
    
    report = {
        "temperature": T,
        "threshold": threshold,
        "splits": {}
    }
    
    # === Evaluate ID splits ===
    for split in ["val", "test_id", "test_ood_chan"]:
        logits, y = load_logits(run_dir, split)
        if logits is None:
            continue
        
        if verbose:
            print(f"=== {split} ===")
        
        # Calibrated probs
        probs_uncal = softmax(logits, 1.0)
        probs_cal = softmax(logits, T)
        
        # Predictions
        preds = np.argmax(probs_cal, axis=1)
        confidences = np.max(probs_cal, axis=1)
        correct = (preds == y)
        
        # Accuracy (ID classes only)
        accuracy = float(np.mean(correct))
        
        # ECE before/after calibration
        ece_before, _ = compute_ece(probs_uncal, y)
        ece_after, bin_data = compute_ece(probs_cal, y)
        
        # Risk-coverage
        rc = compute_risk_coverage(confidences, correct)
        auc_rc = compute_auc_risk_coverage(rc["coverages"], rc["risks"])
        
        # Abstention stats
        abs_stats = abstention_stats(confidences, preds, y, threshold)
        
        if verbose:
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  ECE (before): {ece_before:.4f}")
            print(f"  ECE (after):  {ece_after:.4f}")
            print(f"  Abstention rate: {abs_stats['abstention_rate']:.4f}")
            print(f"  Accuracy on accepted: {abs_stats['accuracy_accepted']:.4f}")
            print()
        
        # Save reliability diagram data
        rel_data = reliability_diagram_data(probs_cal, y)
        np.savez_compressed(
            run_dir / f"reliability_{split}.npz",
            **rel_data
        )
        
        # Save risk-coverage data
        np.savez_compressed(
            run_dir / f"risk_coverage_{split}.npz",
            **rc
        )
        
        report["splits"][split] = {
            "n_examples": len(y),
            "accuracy": accuracy,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "auc_risk_coverage": auc_rc,
            "abstention": abs_stats,
        }
    
    # === OOD Detection ===
    if verbose:
        print("=== OOD Detection (test_id vs test_ood_mod) ===")
    
    logits_id, y_id = load_logits(run_dir, "test_id")
    logits_ood, y_ood = load_logits(run_dir, "test_ood_mod")
    
    if logits_id is not None and logits_ood is not None:
        probs_id = softmax(logits_id, T)
        probs_ood = softmax(logits_ood, T)
        
        scores_id = compute_all_scores(logits_id, probs_id, T)
        scores_ood = compute_all_scores(logits_ood, probs_ood, T)
        
        ood_metrics = {}
        for score_name in ["neg_msp", "entropy", "energy"]:
            auroc = compute_auroc(scores_id[score_name], scores_ood[score_name])
            aupr = compute_aupr(scores_id[score_name], scores_ood[score_name])
            
            ood_metrics[score_name] = {
                "auroc": auroc,
                "aupr": aupr,
            }
            
            if verbose:
                print(f"  {score_name}: AUROC={auroc:.4f}, AUPR={aupr:.4f}")
        
        # Abstention on OOD
        conf_ood = np.max(probs_ood, axis=1)
        ood_abstention_rate = float(np.mean(conf_ood < threshold))
        
        if verbose:
            print(f"\n  OOD abstention rate (τ={threshold:.3f}): {ood_abstention_rate:.4f}")
        
        report["ood_detection"] = {
            "id_split": "test_id",
            "ood_split": "test_ood_mod",
            "metrics": ood_metrics,
            "ood_abstention_rate": ood_abstention_rate,
        }
        
        # Save OOD scores for plotting
        np.savez_compressed(
            run_dir / "ood_scores.npz",
            conf_id=np.max(probs_id, axis=1),
            conf_ood=np.max(probs_ood, axis=1),
            entropy_id=scores_id["entropy"],
            entropy_ood=scores_ood["entropy"],
            energy_id=scores_id["energy"],
            energy_ood=scores_ood["energy"],
        )
    
    # Save report
    report_path = run_dir / "assurance_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    if verbose:
        print()
        print(f"Report saved to: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate assurance metrics")
    parser.add_argument(
        "--run_dir", type=Path, default=None,
        help="V0.5 run directory"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V0.5 Assurance Evaluation")
    print("=" * 60)
    print()
    
    if args.run_dir is None:
        args.run_dir = find_latest_run(project_root / "runs" / "v0_5")
    
    print(f"Run directory: {args.run_dir}")
    print()
    
    evaluate_assurance(args.run_dir)
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
