"""
Evaluation metrics for classification.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Any, Optional


def compute_accuracy(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64]
) -> float:
    """Compute classification accuracy."""
    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    num_classes: Optional[int] = None
) -> NDArray[np.int64]:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of classes (inferred if None).
        
    Returns:
        Confusion matrix, shape (num_classes, num_classes).
        Row i, column j = count of true class i predicted as j.
    """
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    return cm


def compute_per_class_accuracy(
    confusion_matrix: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Compute per-class accuracy from confusion matrix."""
    row_sums = confusion_matrix.sum(axis=1)
    diag = np.diag(confusion_matrix)
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(row_sums > 0, diag / row_sums, 0.0)
    return acc


def compute_precision_recall_f1(
    confusion_matrix: NDArray[np.int64]
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute per-class precision, recall, F1.
    
    Returns:
        Dict with 'precision', 'recall', 'f1' arrays.
    """
    num_classes = confusion_matrix.shape[0]
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp  # Column sum minus diagonal
        fn = confusion_matrix[i, :].sum() - tp  # Row sum minus diagonal
        
        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_macro_f1(confusion_matrix: NDArray[np.int64]) -> float:
    """Compute macro-averaged F1 score."""
    prf = compute_precision_recall_f1(confusion_matrix)
    return float(np.mean(prf["f1"]))


def compute_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
        class_names: List of class names.
        
    Returns:
        Dict with accuracy, macro_f1, confusion_matrix, per_class metrics.
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    prf = compute_precision_recall_f1(cm)
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    per_class = {}
    for i, name in enumerate(class_names):
        if i < len(prf["precision"]):
            per_class[name] = {
                "precision": float(prf["precision"][i]),
                "recall": float(prf["recall"][i]),
                "f1": float(prf["f1"][i]),
                "support": int(cm[i, :].sum()),
            }
    
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "macro_f1": compute_macro_f1(cm),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
