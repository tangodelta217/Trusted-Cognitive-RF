"""
Channel impairment functions for synthetic RF dataset.

Implements realistic RF channel effects:
- AWGN (Additive White Gaussian Noise)
- CFO (Carrier Frequency Offset)
- Phase offset
- Gain variation
- Multipath fading (FIR channel)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Optional
from scipy.signal import lfilter


def add_awgn(
    signal: NDArray[np.complex64],
    snr_db: float,
    rng: np.random.Generator
) -> NDArray[np.complex64]:
    """
    Add Additive White Gaussian Noise to achieve target SNR.
    
    Args:
        signal: Input complex signal (assumed unit power).
        snr_db: Target SNR in dB.
        rng: Random generator.
        
    Returns:
        Noisy signal.
    """
    # Signal power (should be ~1 if normalized)
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # Noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate complex noise
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 
        1j * rng.standard_normal(len(signal))
    )
    
    return (signal + noise).astype(np.complex64)


def apply_cfo(
    signal: NDArray[np.complex64],
    cfo_hz: float,
    fs_hz: float
) -> NDArray[np.complex64]:
    """
    Apply Carrier Frequency Offset.
    
    Args:
        signal: Input complex signal.
        cfo_hz: Frequency offset in Hz.
        fs_hz: Sample rate in Hz.
        
    Returns:
        Signal with frequency offset.
    """
    n = len(signal)
    t = np.arange(n) / fs_hz
    rotation = np.exp(2j * np.pi * cfo_hz * t)
    return (signal * rotation).astype(np.complex64)


def apply_phase_offset(
    signal: NDArray[np.complex64],
    phase_rad: float
) -> NDArray[np.complex64]:
    """
    Apply constant phase offset.
    
    Args:
        signal: Input complex signal.
        phase_rad: Phase offset in radians.
        
    Returns:
        Phase-rotated signal.
    """
    return (signal * np.exp(1j * phase_rad)).astype(np.complex64)


def apply_gain(
    signal: NDArray[np.complex64],
    gain: float
) -> NDArray[np.complex64]:
    """
    Apply gain/attenuation.
    
    Args:
        signal: Input complex signal.
        gain: Gain factor (1.0 = unity).
        
    Returns:
        Scaled signal.
    """
    return (signal * gain).astype(np.complex64)


def apply_multipath(
    signal: NDArray[np.complex64],
    n_taps: int,
    max_delay_samples: int,
    rng: np.random.Generator
) -> NDArray[np.complex64]:
    """
    Apply multipath fading channel.
    
    Generates a random sparse FIR channel with exponential decay.
    
    Args:
        signal: Input complex signal.
        n_taps: Number of channel taps.
        max_delay_samples: Maximum delay spread in samples.
        rng: Random generator.
        
    Returns:
        Signal after multipath channel.
    """
    if n_taps < 1:
        return signal
    
    # Generate tap delays (sorted, first tap at 0)
    if n_taps == 1:
        delays = np.array([0])
    else:
        delays = np.sort(rng.integers(0, max_delay_samples + 1, n_taps))
        delays[0] = 0  # First tap always at 0
    
    # Generate tap coefficients with exponential decay
    decay_rate = 3.0 / max_delay_samples if max_delay_samples > 0 else 0
    magnitudes = np.exp(-decay_rate * delays)
    phases = rng.uniform(0, 2 * np.pi, n_taps)
    coefficients = magnitudes * np.exp(1j * phases)
    
    # Normalize to preserve energy
    coefficients = coefficients / np.sqrt(np.sum(np.abs(coefficients) ** 2))
    
    # Build FIR filter
    h_len = int(max(delays)) + 1
    h = np.zeros(h_len, dtype=np.complex64)
    for delay, coef in zip(delays, coefficients):
        h[int(delay)] += coef
    
    # Apply filter
    output = lfilter(h, 1, signal)
    
    return output.astype(np.complex64)


def apply_impairments(
    signal: NDArray[np.complex64],
    impairment_params: Dict[str, Any],
    fs_hz: float,
    rng: np.random.Generator
) -> NDArray[np.complex64]:
    """
    Apply all channel impairments to a signal.
    
    Args:
        signal: Input complex signal (should be normalized to unit power).
        impairment_params: Dict with sampled impairment values:
            - snr_db: float
            - cfo_hz: float
            - phase_rad: float
            - gain: float
            - multipath_taps: int (optional)
            - multipath_max_delay: int (optional)
        fs_hz: Sample rate in Hz.
        rng: Random generator.
        
    Returns:
        Impaired signal.
    """
    # Apply impairments in order
    
    # 1. Multipath (before noise, as it's part of the channel)
    if impairment_params.get("multipath_taps", 0) > 0:
        signal = apply_multipath(
            signal,
            n_taps=impairment_params["multipath_taps"],
            max_delay_samples=impairment_params.get("multipath_max_delay", 8),
            rng=rng
        )
    
    # 2. CFO
    if "cfo_hz" in impairment_params:
        signal = apply_cfo(signal, impairment_params["cfo_hz"], fs_hz)
    
    # 3. Phase offset
    if "phase_rad" in impairment_params:
        signal = apply_phase_offset(signal, impairment_params["phase_rad"])
    
    # 4. Gain
    if "gain" in impairment_params:
        signal = apply_gain(signal, impairment_params["gain"])
    
    # 5. Renormalize before AWGN for correct SNR
    power = np.mean(np.abs(signal) ** 2)
    if power > 0:
        signal = signal / np.sqrt(power)
    
    # 6. AWGN (last, after power normalization)
    if "snr_db" in impairment_params:
        signal = add_awgn(signal, impairment_params["snr_db"], rng)
    
    return signal


def sample_impairments(
    impairment_config: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Sample random impairment values from configuration ranges.
    
    Args:
        impairment_config: Configuration dict with ranges:
            - snr_db: [min, max]
            - cfo_hz: [min, max]
            - gain: [min, max]
            - phase_rad: [min, max]
            - multipath: {enabled, taps_min, taps_max, max_delay_samples}
        rng: Random generator.
        
    Returns:
        Dict with sampled impairment values.
    """
    params = {}
    
    # Sample SNR
    if "snr_db" in impairment_config:
        snr_range = impairment_config["snr_db"]
        params["snr_db"] = rng.uniform(snr_range[0], snr_range[1])
    
    # Sample CFO
    if "cfo_hz" in impairment_config:
        cfo_range = impairment_config["cfo_hz"]
        params["cfo_hz"] = rng.uniform(cfo_range[0], cfo_range[1])
    
    # Sample gain
    if "gain" in impairment_config:
        gain_range = impairment_config["gain"]
        params["gain"] = rng.uniform(gain_range[0], gain_range[1])
    
    # Sample phase
    if "phase_rad" in impairment_config:
        phase_range = impairment_config["phase_rad"]
        params["phase_rad"] = rng.uniform(phase_range[0], phase_range[1])
    
    # Sample multipath
    multipath_cfg = impairment_config.get("multipath", {})
    if multipath_cfg.get("enabled", False):
        taps_min = multipath_cfg.get("taps_min", 1)
        taps_max = multipath_cfg.get("taps_max", 2)
        params["multipath_taps"] = rng.integers(taps_min, taps_max + 1)
        params["multipath_max_delay"] = multipath_cfg.get("max_delay_samples", 8)
    else:
        params["multipath_taps"] = 0
    
    return params
