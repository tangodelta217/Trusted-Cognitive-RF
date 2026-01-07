"""
Preprocessing functions for IQ signals.

Implements deterministic DC removal and RMS normalization.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any


def remove_dc(iq: NDArray[np.complex64]) -> NDArray[np.complex64]:
    """
    Remove DC offset from IQ signal.
    
    Args:
        iq: Complex IQ signal.
        
    Returns:
        DC-removed signal.
    """
    return (iq - np.mean(iq)).astype(np.complex64)


def rms_normalize(
    iq: NDArray[np.complex64],
    eps: float = 1e-8
) -> NDArray[np.complex64]:
    """
    Normalize IQ signal to unit RMS power.
    
    Args:
        iq: Complex IQ signal.
        eps: Small constant to avoid division by zero.
        
    Returns:
        RMS-normalized signal.
    """
    rms = np.sqrt(np.mean(np.abs(iq) ** 2))
    return (iq / (rms + eps)).astype(np.complex64)


def preprocess(
    iq: NDArray[np.complex64],
    config: Dict[str, Any]
) -> NDArray[np.complex64]:
    """
    Apply preprocessing pipeline to IQ signal.
    
    Args:
        iq: Input IQ signal, shape (N,) complex64.
        config: Preprocessing config dict with keys:
            - remove_dc: bool
            - rms_normalize: bool
            - eps: float
            
    Returns:
        Preprocessed IQ signal.
    """
    # Ensure complex64
    iq = np.asarray(iq, dtype=np.complex64)
    
    # DC removal
    if config.get("remove_dc", True):
        iq = remove_dc(iq)
    
    # RMS normalization
    if config.get("rms_normalize", True):
        eps = config.get("eps", 1e-8)
        iq = rms_normalize(iq, eps)
    
    return iq
