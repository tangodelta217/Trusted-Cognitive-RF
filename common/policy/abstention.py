"""
Abstention policy for selective prediction.

Provides threshold fitting and application for UNKNOWN detection.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, Any


# Operating point presets
PRESETS = {
    "SURVEILLANCE": {
        "coverage_target": 0.95,
        "description": "High coverage, minimize abstentions. For monitoring mode.",
    },
    "TRUSTED": {
        "coverage_target": 0.80,
        "description": "Higher trust, reject more unknowns. For critical decisions.",
    },
    "CONSERVATIVE": {
        "coverage_target": 0.70,
        "description": "Very selective, high accuracy on accepted. For high-stakes.",
    },
}


def fit_threshold_by_coverage(
    conf_scores: NDArray[np.float32],
    target_coverage: float
) -> float:
    """
    Fit confidence threshold for target coverage.
    
    Args:
        conf_scores: Confidence scores (max softmax prob), shape (N,).
        target_coverage: Target fraction of samples to accept (0-1).
        
    Returns:
        Threshold tau such that ~target_coverage samples have conf >= tau.
    """
    if target_coverage <= 0:
        return float(np.max(conf_scores)) + 0.01
    if target_coverage >= 1:
        return 0.0
    
    # Threshold = (1 - target_coverage) percentile
    tau = float(np.percentile(conf_scores, (1 - target_coverage) * 100))
    return tau


def apply_abstention(
    conf_scores: NDArray[np.float32],
    tau: float
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Apply abstention threshold.
    
    Args:
        conf_scores: Confidence scores, shape (N,).
        tau: Confidence threshold.
        
    Returns:
        (accepted_mask, unknown_mask) boolean arrays.
    """
    accepted_mask = conf_scores >= tau
    unknown_mask = ~accepted_mask
    return accepted_mask, unknown_mask


def evaluate_operating_point(
    conf_val: NDArray[np.float32],
    conf_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
    preds_test: NDArray[np.int64],
    conf_ood: NDArray[np.float32],
    target_coverage: float
) -> Dict[str, Any]:
    """
    Evaluate a single operating point.
    
    Args:
        conf_val: Validation confidences (for fitting tau).
        conf_test: Test confidences.
        y_test: True labels for test.
        preds_test: Predictions for test.
        conf_ood: OOD sample confidences.
        target_coverage: Target coverage.
        
    Returns:
        Dict with metrics.
    """
    # Fit threshold on validation
    tau = fit_threshold_by_coverage(conf_val, target_coverage)
    
    # Apply to test
    accepted, unknown = apply_abstention(conf_test, tau)
    
    coverage = float(np.mean(accepted))
    abstention_rate = float(np.mean(unknown))
    
    # Accuracy on accepted
    if accepted.sum() > 0:
        correct_accepted = (preds_test[accepted] == y_test[accepted])
        acc_accepted = float(np.mean(correct_accepted))
    else:
        acc_accepted = 0.0
    
    # Overall accuracy (for reference)
    acc_overall = float(np.mean(preds_test == y_test))
    
    # OOD rejection rate
    ood_rejected, _ = apply_abstention(conf_ood, tau)
    ood_rejection = float(np.mean(~ood_rejected))  # Fraction with conf < tau
    
    return {
        "coverage_target": target_coverage,
        "tau": tau,
        "coverage_actual": coverage,
        "abstention_rate": abstention_rate,
        "accuracy_overall": acc_overall,
        "accuracy_accepted": acc_accepted,
        "ood_rejection": ood_rejection,
    }


def sweep_operating_points(
    conf_val: NDArray[np.float32],
    conf_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
    preds_test: NDArray[np.int64],
    conf_ood: NDArray[np.float32],
    coverage_targets: list = None
) -> list:
    """
    Sweep over multiple operating points.
    
    Args:
        conf_val: Validation confidences.
        conf_test: Test confidences.
        y_test: Test true labels.
        preds_test: Test predictions.
        conf_ood: OOD confidences.
        coverage_targets: List of coverage targets to sweep.
        
    Returns:
        List of dicts with metrics per operating point.
    """
    if coverage_targets is None:
        coverage_targets = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    
    results = []
    for target in coverage_targets:
        result = evaluate_operating_point(
            conf_val, conf_test, y_test, preds_test, conf_ood, target
        )
        results.append(result)
    
    return results
