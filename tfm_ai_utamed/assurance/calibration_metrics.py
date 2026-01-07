"""
Calibration metrics: ECE, reliability diagram.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple


def compute_ece(
    probs: NDArray[np.float32],
    labels: NDArray[np.int64],
    n_bins: int = 15
) -> Tuple[float, Dict[str, NDArray]]:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities, shape (N, C).
        labels: True labels, shape (N,).
        n_bins: Number of confidence bins.
        
    Returns:
        (ece, bin_data) where bin_data contains per-bin statistics.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    ece = 0.0
    n = len(labels)
    
    for lower, upper in zip(bin_lowers, bin_uppers):
        if lower == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        
        count = in_bin.sum()
        bin_counts.append(count)
        
        if count > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            bin_confidences.append(avg_conf)
            bin_accuracies.append(avg_acc)
            ece += (count / n) * abs(avg_acc - avg_conf)
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)
    
    bin_data = {
        "bin_lowers": bin_lowers,
        "bin_uppers": bin_uppers,
        "bin_confidences": np.array(bin_confidences),
        "bin_accuracies": np.array(bin_accuracies),
        "bin_counts": np.array(bin_counts),
    }
    
    return float(ece), bin_data


def reliability_diagram_data(
    probs: NDArray[np.float32],
    labels: NDArray[np.int64],
    n_bins: int = 15
) -> Dict[str, NDArray]:
    """
    Get data for reliability diagram.
    
    Returns dict with bin_centers, bin_accuracies, bin_confidences, bin_counts.
    """
    _, bin_data = compute_ece(probs, labels, n_bins)
    
    bin_centers = (bin_data["bin_lowers"] + bin_data["bin_uppers"]) / 2
    
    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_data["bin_accuracies"],
        "bin_confidences": bin_data["bin_confidences"],
        "bin_counts": bin_data["bin_counts"],
    }


def compute_brier_score(
    probs: NDArray[np.float32],
    labels: NDArray[np.int64]
) -> float:
    """Compute Brier score (multi-class)."""
    n, c = probs.shape
    one_hot = np.zeros((n, c), dtype=np.float32)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
