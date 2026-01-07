"""
Risk-coverage analysis for selective prediction.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple


def compute_risk_coverage(
    confidences: NDArray[np.float32],
    correct: NDArray[np.bool_],
    n_thresholds: int = 100
) -> Dict[str, NDArray]:
    """
    Compute risk-coverage curve.
    
    Risk = error rate on accepted samples.
    Coverage = fraction of samples accepted.
    
    Args:
        confidences: Confidence scores, shape (N,).
        correct: Boolean array of correct predictions, shape (N,).
        n_thresholds: Number of threshold points.
        
    Returns:
        Dict with coverages, risks, thresholds, accuracies.
    """
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


def compute_auc_risk_coverage(
    coverages: NDArray[np.float32],
    risks: NDArray[np.float32]
) -> float:
    """
    Compute AUC of risk-coverage curve.
    
    Lower is better (less risk for same coverage).
    """
    # Sort by coverage
    sorted_indices = np.argsort(coverages)
    coverages_sorted = coverages[sorted_indices]
    risks_sorted = risks[sorted_indices]
    
    # Manual trapezoidal integration (numpy 2.x compatible)
    auc = 0.0
    for i in range(1, len(coverages_sorted)):
        dx = coverages_sorted[i] - coverages_sorted[i-1]
        auc += 0.5 * (risks_sorted[i] + risks_sorted[i-1]) * dx
    
    return float(auc)


def find_coverage_at_target_accuracy(
    accuracies: NDArray[np.float32],
    coverages: NDArray[np.float32],
    target_accuracy: float = 0.90
) -> Tuple[float, float]:
    """
    Find coverage achievable at target accuracy.
    
    Returns:
        (coverage, actual_accuracy) at threshold that first achieves target.
    """
    # Find first threshold where accuracy >= target
    for i, acc in enumerate(accuracies):
        if acc >= target_accuracy:
            return float(coverages[i]), float(acc)
    
    # If never achieved, return lowest coverage
    return 0.0, float(accuracies[-1]) if len(accuracies) > 0 else 0.0


def abstention_stats(
    confidences: NDArray[np.float32],
    predictions: NDArray[np.int64],
    labels: NDArray[np.int64],
    threshold: float
) -> Dict[str, float]:
    """
    Compute statistics for abstention at given threshold.
    
    Args:
        confidences: Confidence scores.
        predictions: Model predictions.
        labels: True labels.
        threshold: Confidence threshold for acceptance.
        
    Returns:
        Dict with coverage, accuracy on accepted, abstention_rate.
    """
    accepted = confidences >= threshold
    n_total = len(labels)
    n_accepted = accepted.sum()
    
    coverage = n_accepted / n_total if n_total > 0 else 0.0
    abstention_rate = 1 - coverage
    
    if n_accepted > 0:
        accuracy_accepted = np.mean(predictions[accepted] == labels[accepted])
    else:
        accuracy_accepted = 0.0
    
    # Overall accuracy if we count abstentions as errors
    accuracy_with_abstention = np.mean(predictions == labels)
    
    return {
        "threshold": threshold,
        "coverage": float(coverage),
        "abstention_rate": float(abstention_rate),
        "accuracy_accepted": float(accuracy_accepted),
        "accuracy_overall": float(accuracy_with_abstention),
        "n_accepted": int(n_accepted),
        "n_abstained": int(n_total - n_accepted),
    }
