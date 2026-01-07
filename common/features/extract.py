"""
Public API for feature extraction.

Provides the main interface for extracting features from IQ signals.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from .preprocess import preprocess
from .stft import stft_from_config
from .spectrogram import spectrogram_from_config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load feature extraction configuration from YAML file.
    
    Args:
        config_path: Path to config YAML. If None, uses default.
        
    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "features_v0.yaml"
    
    config_path = Path(config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_features(
    iq: NDArray[np.complex64],
    config: Optional[Dict[str, Any]] = None
) -> NDArray[np.float32]:
    """
    Extract features from a single IQ signal.
    
    Pipeline:
    1. Preprocess (DC removal, RMS normalize)
    2. STFT (windowed FFT)
    3. Spectrogram (power, log compression)
    4. Normalize (per-example standardization)
    5. Reshape to CHW layout
    
    Args:
        iq: Complex IQ signal, shape (N,) where N=2048.
        config: Feature extraction config. If None, loads default.
        
    Returns:
        Feature tensor, shape (C, F, T) = (1, 256, 15) float32.
    """
    # Load default config if not provided
    if config is None:
        config = load_config()
    
    # Ensure correct input type
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    
    # Validate input length
    expected_samples = config.get("input", {}).get("samples_per_example", 2048)
    if len(iq) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, got {len(iq)}"
        )
    
    # 1. Preprocess
    preprocess_cfg = config.get("preprocess", {})
    iq_preprocessed = preprocess(iq, preprocess_cfg)
    
    # 2. STFT
    stft_cfg = config.get("stft", {})
    stft_out = stft_from_config(iq_preprocessed, stft_cfg)
    
    # 3. Spectrogram with normalization
    spec_cfg = config.get("spectrogram", {})
    norm_cfg = config.get("normalize", {})
    features = spectrogram_from_config(stft_out, spec_cfg, norm_cfg)
    
    # 4. Reshape to CHW layout
    output_cfg = config.get("output", {})
    n_channels = output_cfg.get("channels", 1)
    
    # features is (F, T), reshape to (C, F, T)
    features = features.reshape(n_channels, features.shape[0], features.shape[1])
    
    # Ensure float32
    features = features.astype(np.float32)
    
    # Validate no NaNs
    if np.any(np.isnan(features)):
        raise ValueError("NaN detected in extracted features")
    
    if np.any(np.isinf(features)):
        raise ValueError("Inf detected in extracted features")
    
    return features


def extract_batch(
    iq_batch: NDArray[np.complex64],
    config: Optional[Dict[str, Any]] = None
) -> NDArray[np.float32]:
    """
    Extract features from a batch of IQ signals.
    
    Args:
        iq_batch: Batch of IQ signals, shape (B, N) complex64.
        config: Feature extraction config.
        
    Returns:
        Batch of features, shape (B, C, F, T) float32.
    """
    if config is None:
        config = load_config()
    
    iq_batch = np.asarray(iq_batch, dtype=np.complex64)
    
    if iq_batch.ndim == 1:
        # Single example
        return extract_features(iq_batch, config)[np.newaxis, ...]
    
    batch_size = iq_batch.shape[0]
    
    # Extract first example to get output shape
    first_features = extract_features(iq_batch[0], config)
    output_shape = (batch_size,) + first_features.shape
    
    # Allocate output
    features_batch = np.zeros(output_shape, dtype=np.float32)
    features_batch[0] = first_features
    
    # Extract remaining
    for i in range(1, batch_size):
        features_batch[i] = extract_features(iq_batch[i], config)
    
    return features_batch


def get_output_shape(config: Optional[Dict[str, Any]] = None) -> tuple:
    """
    Get the expected output shape for a single example.
    
    Args:
        config: Feature extraction config.
        
    Returns:
        Tuple (C, F, T).
    """
    if config is None:
        config = load_config()
    
    n_samples = config.get("input", {}).get("samples_per_example", 2048)
    n_fft = config.get("stft", {}).get("n_fft", 256)
    hop = config.get("stft", {}).get("hop", 128)
    n_channels = config.get("output", {}).get("channels", 1)
    
    n_frames = (n_samples - n_fft) // hop + 1
    
    return (n_channels, n_fft, n_frames)
