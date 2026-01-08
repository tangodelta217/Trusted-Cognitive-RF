"""
Temperature Scaling for calibration.

Learns a single temperature T > 0 that rescales logits
to produce better-calibrated probabilities.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Tuple
from scipy.optimize import minimize_scalar


def softmax(logits: NDArray[np.float32], temperature: float = 1.0) -> NDArray[np.float32]:
    """Apply softmax with temperature scaling."""
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return (exp_scaled / exp_scaled.sum(axis=1, keepdims=True)).astype(np.float32)


def nll_loss(probs: NDArray[np.float32], labels: NDArray[np.int64]) -> float:
    """Compute negative log likelihood loss."""
    n = len(labels)
    # HACK: clip para evitar log(0), revisar si hay mejor forma
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    return -np.sum(np.log(probs_clipped[np.arange(n), labels])) / n


class TemperatureScaling:
    """
    Temperature scaling for post-hoc calibration.
    
    Learns T > 0 to minimize NLL on validation set.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.nll_before = None
        self.nll_after = None
    
    def fit(
        self,
        logits: NDArray[np.float32],
        labels: NDArray[np.int64],
        t_min: float = 0.1,
        t_max: float = 10.0
    ) -> float:
        """
        Fit temperature on validation logits.
        
        Args:
            logits: Validation logits, shape (N, C).
            labels: True labels, shape (N,).
            t_min: Minimum temperature to search.
            t_max: Maximum temperature to search.
            
        Returns:
            Optimal temperature.
        """
        # NLL before calibration
        probs_before = softmax(logits, 1.0)
        self.nll_before = nll_loss(probs_before, labels)
        
        # Objective: NLL as function of T
        def objective(t):
            probs = softmax(logits, t)
            return nll_loss(probs, labels)
        
        # Optimize
        result = minimize_scalar(objective, bounds=(t_min, t_max), method='bounded')
        self.temperature = float(result.x)
        
        # NLL after calibration
        probs_after = softmax(logits, self.temperature)
        self.nll_after = nll_loss(probs_after, labels)
        
        return self.temperature
    
    def calibrate(self, logits: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply temperature scaling to logits."""
        return softmax(logits, self.temperature)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dict for JSON serialization."""
        return {
            "temperature": float(self.temperature),
            "nll_before": float(self.nll_before) if self.nll_before is not None else None,
            "nll_after": float(self.nll_after) if self.nll_after is not None else None,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemperatureScaling":
        """Load from dict."""
        ts = cls()
        ts.temperature = d["temperature"]
        ts.nll_before = d.get("nll_before")
        ts.nll_after = d.get("nll_after")
        return ts


def find_threshold_for_coverage(
    confidences: NDArray[np.float32],
    target_coverage: float = 0.95
) -> Tuple[float, float]:
    """
    Find confidence threshold for target coverage.
    
    Args:
        confidences: Max softmax probabilities, shape (N,).
        target_coverage: Target fraction of samples to accept.
        
    Returns:
        (threshold, actual_coverage)
    """
    # Threshold = (1 - target_coverage) percentile
    threshold = float(np.percentile(confidences, (1 - target_coverage) * 100))
    actual_coverage = float(np.mean(confidences >= threshold))
    return threshold, actual_coverage
