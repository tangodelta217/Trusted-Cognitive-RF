#!/usr/bin/env python3
"""
Calibrate abstention thresholds for UNKNOWN detection.

Given coverage targets for each mode, finds τ values such that:
- max_prob < τ => UNKNOWN

Usage:
    python -m tools.calibrate_thresholds
    python -m tools.calibrate_thresholds --coverage_surv 0.95 --coverage_trusted 0.85 --coverage_cons 0.75
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
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


def load_calibrated_probs(run_dir: Path, split: str = "val") -> Tuple[np.ndarray, np.ndarray]:
    """Load calibrated probabilities for a split."""
    from tfm_ai_utamed.assurance.temperature_scaling import softmax
    
    logits_path = run_dir / f"logits_{split}.npz"
    if not logits_path.exists():
        return None, None
    
    data = np.load(logits_path)
    logits = data["logits"]
    labels = data["y"]
    
    # Load temperature
    temp_path = run_dir / "temperature.json"
    if temp_path.exists():
        with open(temp_path) as f:
            T = json.load(f)["temperature"]
    else:
        # Try artifacts
        artifacts_temp = project_root / "artifacts" / "policy" / "temperature.json"
        if artifacts_temp.exists():
            with open(artifacts_temp) as f:
                T = json.load(f)["temperature"]
        else:
            T = 1.0
    
    probs = softmax(logits.astype(np.float32), T)
    return probs, labels


def find_threshold_for_coverage(confidences: np.ndarray, target_coverage: float) -> Tuple[float, float]:
    """
    Find threshold τ such that coverage ≈ target_coverage.
    
    Coverage = fraction of samples where max_prob >= τ.
    
    Returns:
        (threshold, actual_coverage)
    """
    # τ is the (1 - target_coverage) percentile of confidences
    percentile = (1 - target_coverage) * 100
    threshold = float(np.percentile(confidences, percentile))
    actual_coverage = float(np.mean(confidences >= threshold))
    return threshold, actual_coverage


def compute_abstention_stats(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute statistics for a given threshold."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    accepted = confidences >= threshold
    n_total = len(labels)
    n_accepted = accepted.sum()
    
    coverage = n_accepted / n_total
    abstention_rate = 1 - coverage
    
    if n_accepted > 0:
        accuracy_accepted = float(np.mean(predictions[accepted] == labels[accepted]))
        risk = 1 - accuracy_accepted  # Error rate on accepted
    else:
        accuracy_accepted = 0.0
        risk = 0.0
    
    return {
        "threshold": threshold,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "accuracy_accepted": accuracy_accepted,
        "risk": risk,
        "n_accepted": int(n_accepted),
        "n_abstained": int(n_total - n_accepted),
    }


def compute_risk_coverage_curve(probs: np.ndarray, labels: np.ndarray, 
                                 n_thresholds: int = 100) -> Dict[str, np.ndarray]:
    """Compute risk-coverage curve."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)
    
    thresholds = np.linspace(0, 1, n_thresholds)
    
    coverages = []
    risks = []
    accuracies = []
    
    for tau in thresholds:
        accepted = confidences >= tau
        coverage = np.mean(accepted)
        
        if coverage > 0:
            risk = 1 - np.mean(correct[accepted])
            accuracy = np.mean(correct[accepted])
        else:
            risk = 0.0
            accuracy = 0.0
        
        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(accuracy)
    
    return {
        "thresholds": thresholds,
        "coverages": np.array(coverages),
        "risks": np.array(risks),
        "accuracies": np.array(accuracies),
    }


def plot_risk_coverage(rc_data: Dict, thresholds_info: Dict, save_path: Path) -> None:
    """Generate and save risk-coverage plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    coverages = rc_data["coverages"]
    risks = rc_data["risks"]
    
    # Main curve
    ax.plot(coverages, risks, 'b-', linewidth=2, label='Risk-Coverage')
    ax.fill_between(coverages, risks, alpha=0.2)
    
    # Mark operating points
    colors = {'SURVEILLANCE': 'green', 'TRUSTED': 'orange', 'CONSERVATIVE': 'red'}
    for mode, info in thresholds_info["modes"].items():
        cov = info["coverage"]
        # Find risk at this coverage
        idx = np.argmin(np.abs(coverages - cov))
        risk = risks[idx]
        ax.scatter([cov], [risk], s=100, c=colors.get(mode, 'gray'), 
                   label=f'{mode} (τ={info["tau"]:.3f})', zorder=5)
        ax.annotate(mode[:4], (cov, risk), textcoords="offset points", 
                    xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel("Coverage (fraction accepted)", fontsize=12)
    ax.set_ylabel("Risk (error rate on accepted)", fontsize=12)
    ax.set_title("Risk-Coverage Curve with Operating Points", fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.3, np.max(risks) * 1.2))
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate abstention thresholds for 3 operating modes"
    )
    parser.add_argument("--coverage_surv", type=float, default=0.95,
                        help="Target coverage for SURVEILLANCE mode")
    parser.add_argument("--coverage_trusted", type=float, default=0.85,
                        help="Target coverage for TRUSTED mode")
    parser.add_argument("--coverage_cons", type=float, default=0.75,
                        help="Target coverage for CONSERVATIVE mode")
    parser.add_argument("--run_dir", type=Path, default=None,
                        help="Run directory with logits")
    
    args = parser.parse_args()
    
    # buscar directorio con logits
    if args.run_dir is None:
        args.run_dir = find_latest_logits_run()
        if args.run_dir is None:
            print("Error: no hay runs con logits")
            sys.exit(1)
    
    print(f"Dir: {args.run_dir}")
    
    # cargar datos val
    probs, labels = load_calibrated_probs(args.run_dir, "val")
    if probs is None:
        print("Error: no val data")
        sys.exit(1)
    
    print(f"{len(labels)} samples")
    
    confidences = np.max(probs, axis=1)
    
    # calc thresholds
    print("Calculando...")
    
    modes = {
        "SURVEILLANCE": args.coverage_surv,
        "TRUSTED": args.coverage_trusted,
        "CONSERVATIVE": args.coverage_cons,
    }
    
    thresholds_info = {
        "modes": {},
        "calibrated_on": "val",
        "n_samples": len(labels),
    }
    
    for mode, target_cov in modes.items():
        tau, actual_cov = find_threshold_for_coverage(confidences, target_cov)
        stats = compute_abstention_stats(probs, labels, tau)
        
        thresholds_info["modes"][mode] = {
            "tau": round(tau, 6),
            "target_coverage": target_cov,
            "coverage": round(actual_cov, 4),
            "accuracy_accepted": round(stats["accuracy_accepted"], 4),
            "risk": round(stats["risk"], 4),
            "abstention_rate": round(stats["abstention_rate"], 4),
        }
        
        print(f"  {mode}:")
        print(f"    τ = {tau:.4f}")
        print(f"    coverage = {actual_cov:.4f} (target: {target_cov:.4f})")
        print(f"    accuracy on accepted = {stats['accuracy_accepted']:.4f}")
        print(f"    risk (error rate) = {stats['risk']:.4f}")
    
    # Verify ordering: τ_cons >= τ_trusted >= τ_surv
    taus = [thresholds_info["modes"][m]["tau"] for m in ["SURVEILLANCE", "TRUSTED", "CONSERVATIVE"]]
    if not (taus[0] <= taus[1] <= taus[2]):
        print("\nWARNING: Threshold ordering not as expected!")
    else:
        print(f"\n✓ Threshold ordering verified: τ_SURV({taus[0]:.4f}) <= τ_TRUSTED({taus[1]:.4f}) <= τ_CONS({taus[2]:.4f})")
    
    # Save thresholds
    artifacts_dir = project_root / "artifacts" / "policy"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = artifacts_dir / "thresholds.json"
    
    with open(thresholds_path, "w") as f:
        json.dump(thresholds_info, f, indent=2)
    
    print(f"\nSaved: {thresholds_path}")
    
    # Compute and save risk-coverage curve
    print("\nGenerating risk-coverage curve...")
    rc_data = compute_risk_coverage_curve(probs, labels)
    
    reports_dir = project_root / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rc_path = reports_dir / "risk_coverage.png"
    
    plot_risk_coverage(rc_data, thresholds_info, rc_path)
    print(f"Saved: {rc_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
