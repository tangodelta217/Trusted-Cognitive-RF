"""
Spectrogram computation from STFT output.

Converts complex STFT to log-power spectrogram.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any


def compute_power_spectrogram(
    stft: NDArray[np.complex64]
) -> NDArray[np.float32]:
    """
    Compute power spectrogram from STFT.
    
    Args:
        stft: Complex STFT output, shape (F, T).
        
    Returns:
        Power spectrogram |X|^2, shape (F, T) float32.
    """
    return (np.abs(stft) ** 2).astype(np.float32)


def compute_log_spectrogram(
    power: NDArray[np.float32],
    log_fn: str = "log1p"
) -> NDArray[np.float32]:
    """
    Apply log compression to power spectrogram.
    
    Args:
        power: Power spectrogram.
        log_fn: Log function ('log1p', 'log10', 'log').
        
    Returns:
        Log-compressed spectrogram.
    """
    if log_fn == "log1p":
        return np.log1p(power).astype(np.float32)
    elif log_fn == "log10":
        return np.log10(power + 1e-10).astype(np.float32)
    elif log_fn == "log":
        return np.log(power + 1e-10).astype(np.float32)
    else:
        raise ValueError(f"Unknown log function: {log_fn}")


def normalize_spectrogram(
    spec: NDArray[np.float32],
    mode: str = "per_example_standardize",
    eps: float = 1e-8
) -> NDArray[np.float32]:
    """
    Normalize spectrogram.
    
    Args:
        spec: Input spectrogram, shape (F, T).
        mode: Normalization mode:
            - 'per_example_standardize': (S - mean) / std
            - 'per_example_minmax': (S - min) / (max - min)
            - 'none': no normalization
        eps: Small constant for numerical stability.
        
    Returns:
        Normalized spectrogram.
    """
    if mode == "per_example_standardize":
        mean = np.mean(spec)
        std = np.std(spec)
        return ((spec - mean) / (std + eps)).astype(np.float32)
    
    elif mode == "per_example_minmax":
        min_val = np.min(spec)
        max_val = np.max(spec)
        return ((spec - min_val) / (max_val - min_val + eps)).astype(np.float32)
    
    elif mode == "none":
        return spec
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def spectrogram_from_config(
    stft: NDArray[np.complex64],
    spec_config: Dict[str, Any],
    norm_config: Dict[str, Any]
) -> NDArray[np.float32]:
    """
    Compute normalized spectrogram from STFT using configuration.
    
    Args:
        stft: Complex STFT output.
        spec_config: Spectrogram config (power, log, output_dtype).
        norm_config: Normalization config (mode, eps).
        
    Returns:
        Normalized spectrogram, shape (F, T) float32.
    """
    # Power spectrogram
    if spec_config.get("power", True):
        spec = compute_power_spectrogram(stft)
    else:
        spec = np.abs(stft).astype(np.float32)
    
    # Log compression
    log_fn = spec_config.get("log", "log1p")
    if log_fn and log_fn != "none":
        spec = compute_log_spectrogram(spec, log_fn)
    
    # Normalization
    mode = norm_config.get("mode", "per_example_standardize")
    eps = norm_config.get("eps", 1e-8)
    spec = normalize_spectrogram(spec, mode, eps)
    
    return spec
