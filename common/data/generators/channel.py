"""
Combined channel model for synthetic RF dataset.

Provides a unified interface for generating impaired signals.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any

from .modulations import generate_modulated_signal
from .impairments import apply_impairments, sample_impairments


def apply_channel(
    signal: NDArray[np.complex64],
    impairment_config: Dict[str, Any],
    fs_hz: float,
    rng: np.random.Generator
) -> tuple[NDArray[np.complex64], Dict[str, Any]]:
    """
    Apply random channel impairments to a signal.
    
    Args:
        signal: Input complex signal.
        impairment_config: Impairment configuration with ranges.
        fs_hz: Sample rate in Hz.
        rng: Random generator.
        
    Returns:
        Tuple of (impaired_signal, sampled_params).
    """
    # Sample impairment values
    params = sample_impairments(impairment_config, rng)
    
    # Apply impairments
    impaired = apply_impairments(signal, params, fs_hz, rng)
    
    return impaired, params


def generate_example(
    modulation: str,
    n_samples: int,
    sps: int,
    impairment_config: Dict[str, Any],
    fs_hz: float,
    rng: np.random.Generator,
    rrc_rolloff: float = 0.35,
    rrc_span_symbols: int = 8
) -> tuple[NDArray[np.complex64], Dict[str, Any]]:
    """
    Generate a complete example: modulated signal + channel impairments.
    
    Args:
        modulation: Modulation type.
        n_samples: Number of output samples.
        sps: Samples per symbol.
        impairment_config: Impairment configuration.
        fs_hz: Sample rate in Hz.
        rng: Random generator.
        rrc_rolloff: RRC filter roll-off.
        rrc_span_symbols: RRC filter span.
        
    Returns:
        Tuple of (iq_signal, metadata_dict).
    """
    # Generate clean signal
    signal = generate_modulated_signal(
        modulation=modulation,
        n_samples=n_samples,
        sps=sps,
        rng=rng,
        rrc_rolloff=rrc_rolloff,
        rrc_span_symbols=rrc_span_symbols
    )
    
    # Apply channel
    impaired, params = apply_channel(signal, impairment_config, fs_hz, rng)
    
    # Build metadata
    metadata = {
        "modulation": modulation,
        **params
    }
    
    return impaired, metadata
