"""
Inference utilities for model prediction.

Supports both PyTorch and ONNX inference.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn


def load_model(
    checkpoint_path: Path,
    model_config_path: Path,
    device: str = "cpu"
) -> nn.Module:
    """
    Load PyTorch model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint.
        model_config_path: Path to model config YAML.
        device: Device to load model on.
        
    Returns:
        Model in eval mode.
    """
    from common.models.cnn_small import build_model_from_config, load_model_config
    
    model_cfg = load_model_config(model_config_path)
    model = build_model_from_config(model_cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    
    return model


def predict_logits(
    model: nn.Module,
    X: NDArray[np.float32],
    batch_size: int = 64,
    device: str = "cpu"
) -> NDArray[np.float32]:
    """
    Get model logits for input features.
    
    Args:
        model: PyTorch model.
        X: Features, shape (N, C, F, T).
        batch_size: Batch size for inference.
        device: Device to run on.
        
    Returns:
        Logits, shape (N, num_classes).
    """
    model.eval()
    
    n_samples = len(X)
    all_logits = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            logits = model(batch)
            all_logits.append(logits.cpu().numpy())
    
    return np.concatenate(all_logits, axis=0).astype(np.float32)


def predict_probs(
    model: nn.Module,
    X: NDArray[np.float32],
    temperature: float = 1.0,
    batch_size: int = 64,
    device: str = "cpu"
) -> NDArray[np.float32]:
    """
    Get calibrated probabilities.
    
    Args:
        model: PyTorch model.
        X: Features.
        temperature: Temperature for calibration.
        batch_size: Batch size.
        device: Device.
        
    Returns:
        Probabilities, shape (N, num_classes).
    """
    logits = predict_logits(model, X, batch_size, device)
    
    # Apply temperature scaling
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    
    return probs.astype(np.float32)


def predict_with_abstention(
    probs: NDArray[np.float32],
    threshold: float
) -> Tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """
    Make predictions with abstention.
    
    Args:
        probs: Probabilities, shape (N, num_classes).
        threshold: Confidence threshold for acceptance.
        
    Returns:
        (predictions, accepted) where:
            predictions: Predicted class indices (UNKNOWN = -1)
            accepted: Boolean mask of accepted samples
    """
    confidences = np.max(probs, axis=1)
    accepted = confidences >= threshold
    
    predictions = np.argmax(probs, axis=1)
    predictions = predictions.astype(np.int64)
    
    # Mark abstained as -1 (UNKNOWN)
    predictions[~accepted] = -1
    
    return predictions, accepted
