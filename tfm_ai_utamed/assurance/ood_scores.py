"""
OOD detection scores: MSP, entropy, energy.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict


def compute_msp(probs: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Compute Maximum Softmax Probability (MSP).
    
    Higher MSP = more confident = likely ID.
    For OOD detection, use negative MSP as score (higher = more OOD).
    
    Args:
        probs: Softmax probabilities, shape (N, C).
        
    Returns:
        MSP scores, shape (N,). Higher = more confident.
    """
    return np.max(probs, axis=1).astype(np.float32)


def compute_entropy(probs: NDArray[np.float32], eps: float = 1e-10) -> NDArray[np.float32]:
    """
    Compute prediction entropy.
    
    Higher entropy = more uncertain = likely OOD.
    
    Args:
        probs: Softmax probabilities, shape (N, C).
        eps: Small constant for numerical stability.
        
    Returns:
        Entropy scores, shape (N,). Higher = more uncertain.
    """
    log_probs = np.log(probs + eps)
    return (-np.sum(probs * log_probs, axis=1)).astype(np.float32)


def compute_energy(logits: NDArray[np.float32], temperature: float = 1.0) -> NDArray[np.float32]:
    """
    Compute energy score (Liu et al., NeurIPS 2020).
    
    Energy = -T * log(sum(exp(logits/T)))
    Lower energy = more confident = likely ID.
    For OOD detection, use energy (higher = more OOD).
    
    Args:
        logits: Raw logits, shape (N, C).
        temperature: Temperature for scaling.
        
    Returns:
        Energy scores, shape (N,). Higher = more OOD.
    """
    scaled = logits / temperature
    # LogSumExp for numerical stability
    max_logits = np.max(scaled, axis=1, keepdims=True)
    logsumexp = max_logits.squeeze() + np.log(np.sum(np.exp(scaled - max_logits), axis=1))
    return (-temperature * logsumexp).astype(np.float32)


def compute_all_scores(
    logits: NDArray[np.float32],
    probs: NDArray[np.float32],
    temperature: float = 1.0
) -> Dict[str, NDArray[np.float32]]:
    """
    Compute all OOD scores.
    
    Returns dict with:
        - msp: Higher = more confident (ID)
        - neg_msp: Higher = more OOD
        - entropy: Higher = more OOD
        - energy: Higher = more OOD
    """
    msp = compute_msp(probs)
    entropy = compute_entropy(probs)
    energy = compute_energy(logits, temperature)
    
    return {
        "msp": msp,
        "neg_msp": -msp,
        "entropy": entropy,
        "energy": energy,
    }


def compute_auroc(
    scores_id: NDArray[np.float32],
    scores_ood: NDArray[np.float32]
) -> float:
    """
    Compute AUROC for OOD detection.
    
    Convention: higher score = more OOD.
    
    Args:
        scores_id: Scores for ID samples (should be lower).
        scores_ood: Scores for OOD samples (should be higher).
        
    Returns:
        AUROC (Area Under ROC Curve).
    """
    # Labels: 0 = ID, 1 = OOD
    labels = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    scores = np.concatenate([scores_id, scores_ood])
    
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    
    # Compute TPR/FPR at each threshold
    n_pos = len(scores_ood)  # OOD
    n_neg = len(scores_id)   # ID
    
    tp = 0
    fp = 0
    tpr_prev = 0
    fpr_prev = 0
    auroc = 0.0
    
    for i, label in enumerate(sorted_labels):
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        # Trapezoidal rule
        auroc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        
        tpr_prev = tpr
        fpr_prev = fpr
    
    return float(auroc)


def compute_aupr(
    scores_id: NDArray[np.float32],
    scores_ood: NDArray[np.float32]
) -> float:
    """
    Compute AUPR (Area Under Precision-Recall curve) for OOD detection.
    
    Convention: higher score = more OOD (positive class).
    """
    labels = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    scores = np.concatenate([scores_id, scores_ood])
    
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    
    n_pos = len(scores_ood)
    
    tp = 0
    fp = 0
    precision_prev = 1.0
    recall_prev = 0.0
    aupr = 0.0
    
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / n_pos if n_pos > 0 else 0
        
        # Trapezoidal rule
        aupr += (recall - recall_prev) * (precision + precision_prev) / 2
        
        precision_prev = precision
        recall_prev = recall
    
    return float(aupr)
