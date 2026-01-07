"""
Plotting utilities for assurance visualizations.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_reliability_diagram(
    bin_data: Dict[str, NDArray],
    title: str = "Reliability Diagram",
    save_path: Optional[Path] = None
) -> None:
    """
    Plot reliability diagram.
    
    Args:
        bin_data: Dict with bin_centers, bin_accuracies, bin_confidences, bin_counts.
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    bin_centers = bin_data["bin_centers"]
    bin_accuracies = bin_data["bin_accuracies"]
    bin_counts = bin_data["bin_counts"]
    
    # Only plot bins with samples
    mask = bin_counts > 0
    
    # Bar chart
    width = 1.0 / len(bin_centers) * 0.8
    ax.bar(bin_centers[mask], bin_accuracies[mask], width=width, 
           alpha=0.7, label="Accuracy", color="steelblue")
    
    # Diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
    
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_risk_coverage(
    coverages: NDArray[np.float32],
    risks: NDArray[np.float32],
    title: str = "Risk-Coverage Curve",
    save_path: Optional[Path] = None
) -> None:
    """Plot risk-coverage curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.plot(coverages, risks, 'b-', linewidth=2)
    ax.fill_between(coverages, risks, alpha=0.3)
    
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (Error Rate)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.5, np.max(risks) * 1.1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_ood_roc(
    fpr: NDArray[np.float32],
    tpr: NDArray[np.float32],
    auroc: float,
    title: str = "OOD Detection ROC",
    save_path: Optional[Path] = None
) -> None:
    """Plot ROC curve for OOD detection."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f"ROC (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', label="Random")
    
    ax.set_xlabel("False Positive Rate (ID classified as OOD)")
    ax.set_ylabel("True Positive Rate (OOD detected)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def compute_roc_curve(
    scores_id: NDArray[np.float32],
    scores_ood: NDArray[np.float32],
    n_thresholds: int = 200
) -> Dict[str, NDArray]:
    """
    Compute ROC curve data.
    
    Convention: higher score = more OOD.
    """
    all_scores = np.concatenate([scores_id, scores_ood])
    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), n_thresholds)
    
    fprs = []
    tprs = []
    
    n_id = len(scores_id)
    n_ood = len(scores_ood)
    
    for thresh in thresholds:
        # Predict OOD if score >= threshold
        fp = np.sum(scores_id >= thresh)  # ID misclassified as OOD
        tp = np.sum(scores_ood >= thresh)  # OOD correctly detected
        
        fpr = fp / n_id if n_id > 0 else 0
        tpr = tp / n_ood if n_ood > 0 else 0
        
        fprs.append(fpr)
        tprs.append(tpr)
    
    return {
        "fpr": np.array(fprs),
        "tpr": np.array(tprs),
        "thresholds": thresholds,
    }


def plot_confidence_histograms(
    conf_id: NDArray[np.float32],
    conf_ood: NDArray[np.float32],
    threshold: Optional[float] = None,
    title: str = "Confidence Distribution",
    save_path: Optional[Path] = None
) -> None:
    """Plot confidence histograms for ID and OOD."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = np.linspace(0, 1, 30)
    
    ax.hist(conf_id, bins=bins, alpha=0.6, label="ID", color="steelblue", density=True)
    ax.hist(conf_ood, bins=bins, alpha=0.6, label="OOD", color="coral", density=True)
    
    if threshold is not None:
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f"Threshold τ={threshold:.3f}")
    
    ax.set_xlabel("Confidence (MSP)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
