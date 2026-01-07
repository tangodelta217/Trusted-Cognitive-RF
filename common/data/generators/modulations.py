"""
Modulation signal generators for synthetic RF dataset.

Generates baseband IQ signals for various modulation schemes:
- PSK: BPSK, QPSK, 8PSK
- QAM: 16QAM, 64QAM
- FSK: GFSK, CPFSK
- NOISE: Pure noise (idle channel)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Optional
from scipy.signal import firwin, lfilter


def _rrc_filter(sps: int, span_symbols: int, rolloff: float) -> NDArray[np.float64]:
    """
    Generate Root Raised Cosine (RRC) pulse shaping filter.
    
    Args:
        sps: Samples per symbol.
        span_symbols: Filter span in symbols.
        rolloff: Roll-off factor (0 to 1).
        
    Returns:
        RRC filter coefficients.
    """
    N = span_symbols * sps
    t = np.arange(-N // 2, N // 2 + 1) / sps
    
    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 + rolloff * (4 / np.pi - 1)
        elif abs(ti) == 1 / (4 * rolloff) and rolloff != 0:
            h[i] = (rolloff / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            )
        else:
            num = np.sin(np.pi * ti * (1 - rolloff)) + \
                  4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff))
            denom = np.pi * ti * (1 - (4 * rolloff * ti) ** 2)
            if abs(denom) > 1e-10:
                h[i] = num / denom
            else:
                h[i] = 0
    
    # Normalize filter energy
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def _upsample_and_filter(
    symbols: NDArray[np.complex64],
    sps: int,
    rrc_filter: NDArray[np.float64]
) -> NDArray[np.complex64]:
    """
    Upsample symbols and apply pulse shaping filter.
    
    Args:
        symbols: Complex symbol sequence.
        sps: Samples per symbol.
        rrc_filter: Pulse shaping filter.
        
    Returns:
        Upsampled and filtered signal.
    """
    # Upsample by inserting zeros
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    
    # Apply filter
    filtered = lfilter(rrc_filter, 1, upsampled)
    return filtered.astype(np.complex64)


def generate_bpsk(
    n_samples: int,
    sps: int,
    rng: np.random.Generator,
    rrc_filter: Optional[NDArray[np.float64]] = None
) -> NDArray[np.complex64]:
    """Generate BPSK modulated signal."""
    n_symbols = n_samples // sps + 10  # Extra for filter delay
    bits = rng.integers(0, 2, n_symbols)
    symbols = 2 * bits - 1  # Map to {-1, +1}
    symbols = symbols.astype(np.complex64)
    
    if rrc_filter is not None:
        signal = _upsample_and_filter(symbols, sps, rrc_filter)
    else:
        signal = np.repeat(symbols, sps)
    
    return signal[:n_samples].astype(np.complex64)


def generate_qpsk(
    n_samples: int,
    sps: int,
    rng: np.random.Generator,
    rrc_filter: Optional[NDArray[np.float64]] = None
) -> NDArray[np.complex64]:
    """Generate QPSK modulated signal."""
    n_symbols = n_samples // sps + 10
    bits = rng.integers(0, 4, n_symbols)
    
    # Gray coded QPSK constellation
    constellation = np.array([
        1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j
    ], dtype=np.complex64) / np.sqrt(2)
    
    symbols = constellation[bits]
    
    if rrc_filter is not None:
        signal = _upsample_and_filter(symbols, sps, rrc_filter)
    else:
        signal = np.repeat(symbols, sps)
    
    return signal[:n_samples].astype(np.complex64)


def generate_8psk(
    n_samples: int,
    sps: int,
    rng: np.random.Generator,
    rrc_filter: Optional[NDArray[np.float64]] = None
) -> NDArray[np.complex64]:
    """Generate 8PSK modulated signal."""
    n_symbols = n_samples // sps + 10
    bits = rng.integers(0, 8, n_symbols)
    
    # 8PSK constellation
    phases = 2 * np.pi * np.arange(8) / 8
    constellation = np.exp(1j * phases).astype(np.complex64)
    
    symbols = constellation[bits]
    
    if rrc_filter is not None:
        signal = _upsample_and_filter(symbols, sps, rrc_filter)
    else:
        signal = np.repeat(symbols, sps)
    
    return signal[:n_samples].astype(np.complex64)


def generate_16qam(
    n_samples: int,
    sps: int,
    rng: np.random.Generator,
    rrc_filter: Optional[NDArray[np.float64]] = None
) -> NDArray[np.complex64]:
    """Generate 16-QAM modulated signal."""
    n_symbols = n_samples // sps + 10
    bits = rng.integers(0, 16, n_symbols)
    
    # 16-QAM constellation (4x4 grid)
    levels = np.array([-3, -1, 1, 3])
    constellation = np.array([
        levels[i] + 1j * levels[j]
        for i in range(4) for j in range(4)
    ], dtype=np.complex64)
    constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    
    symbols = constellation[bits]
    
    if rrc_filter is not None:
        signal = _upsample_and_filter(symbols, sps, rrc_filter)
    else:
        signal = np.repeat(symbols, sps)
    
    return signal[:n_samples].astype(np.complex64)


def generate_64qam(
    n_samples: int,
    sps: int,
    rng: np.random.Generator,
    rrc_filter: Optional[NDArray[np.float64]] = None
) -> NDArray[np.complex64]:
    """Generate 64-QAM modulated signal."""
    n_symbols = n_samples // sps + 10
    bits = rng.integers(0, 64, n_symbols)
    
    # 64-QAM constellation (8x8 grid)
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    constellation = np.array([
        levels[i] + 1j * levels[j]
        for i in range(8) for j in range(8)
    ], dtype=np.complex64)
    constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    
    symbols = constellation[bits]
    
    if rrc_filter is not None:
        signal = _upsample_and_filter(symbols, sps, rrc_filter)
    else:
        signal = np.repeat(symbols, sps)
    
    return signal[:n_samples].astype(np.complex64)


def generate_gfsk(
    n_samples: int,
    sps: int,
    rng: np.random.Generator,
    rrc_filter: Optional[NDArray[np.float64]] = None,  # Not used, for API consistency
    bt: float = 0.5,
    mod_index: float = 0.5,
) -> NDArray[np.complex64]:
    """
    Generate GFSK modulated signal.
    
    Args:
        n_samples: Number of output samples.
        sps: Samples per symbol.
        rng: Random number generator.
        rrc_filter: Not used (for API consistency).
        bt: Bandwidth-time product (Gaussian filter).
        mod_index: Modulation index (h).
    """
    n_symbols = n_samples // sps + 10
    bits = rng.integers(0, 2, n_symbols)
    nrz = 2 * bits - 1  # NRZ encoding
    
    # Upsample
    nrz_up = np.repeat(nrz, sps)
    
    # Gaussian filter
    span = 4  # symbols
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    alpha = np.sqrt(np.log(2) / 2) / bt
    gaussian = np.sqrt(np.pi) / alpha * np.exp(-(np.pi * t / alpha) ** 2)
    gaussian = gaussian / np.sum(gaussian)
    
    # Filter NRZ signal
    filtered = np.convolve(nrz_up, gaussian, mode='same')
    
    # Integrate to get phase
    phase = np.cumsum(filtered) * np.pi * mod_index / sps
    
    # Generate complex signal
    signal = np.exp(1j * phase).astype(np.complex64)
    
    return signal[:n_samples]


def generate_cpfsk(
    n_samples: int,
    sps: int,
    rng: np.random.Generator,
    rrc_filter: Optional[NDArray[np.float64]] = None,  # Not used, for API consistency
    mod_index: float = 0.5,
) -> NDArray[np.complex64]:
    """
    Generate CPFSK (Continuous Phase FSK) modulated signal.
    
    Args:
        n_samples: Number of output samples.
        sps: Samples per symbol.
        rng: Random number generator.
        rrc_filter: Not used (for API consistency).
        mod_index: Modulation index (h).
    """
    n_symbols = n_samples // sps + 10
    bits = rng.integers(0, 2, n_symbols)
    nrz = 2 * bits - 1  # NRZ encoding
    
    # Upsample
    nrz_up = np.repeat(nrz, sps)
    
    # Rectangular frequency pulse (no Gaussian filtering)
    # Integrate to get phase
    phase = np.cumsum(nrz_up) * np.pi * mod_index / sps
    
    # Generate complex signal
    signal = np.exp(1j * phase).astype(np.complex64)
    
    return signal[:n_samples]


def generate_noise(
    n_samples: int,
    sps: int,
    rng: np.random.Generator,
    rrc_filter: Optional[NDArray[np.float64]] = None
) -> NDArray[np.complex64]:
    """Generate pure complex Gaussian noise (no signal)."""
    noise = (rng.standard_normal(n_samples) + 
             1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    return noise.astype(np.complex64)


# Mapping from modulation name to generator function
MODULATION_MAP: Dict[str, Callable] = {
    "BPSK": generate_bpsk,
    "QPSK": generate_qpsk,
    "PSK8": generate_8psk,
    "QAM16": generate_16qam,
    "QAM64": generate_64qam,
    "GFSK": generate_gfsk,
    "CPFSK": generate_cpfsk,
    "NOISE": generate_noise,
}


def generate_modulated_signal(
    modulation: str,
    n_samples: int,
    sps: int,
    rng: np.random.Generator,
    rrc_rolloff: float = 0.35,
    rrc_span_symbols: int = 8
) -> NDArray[np.complex64]:
    """
    Generate a modulated IQ signal.
    
    Args:
        modulation: Modulation type (e.g., 'BPSK', 'QPSK', 'QAM16').
        n_samples: Number of output samples.
        sps: Samples per symbol.
        rng: NumPy random generator for reproducibility.
        rrc_rolloff: RRC filter roll-off factor.
        rrc_span_symbols: RRC filter span in symbols.
        
    Returns:
        Complex IQ signal as numpy array.
    """
    if modulation not in MODULATION_MAP:
        raise ValueError(f"Unknown modulation: {modulation}. "
                        f"Available: {list(MODULATION_MAP.keys())}")
    
    # Generate RRC filter for linear modulations
    rrc_filter = None
    if modulation in ["BPSK", "QPSK", "PSK8", "QAM16", "QAM64"]:
        rrc_filter = _rrc_filter(sps, rrc_span_symbols, rrc_rolloff)
    
    generator = MODULATION_MAP[modulation]
    signal = generator(n_samples, sps, rng, rrc_filter)
    
    # Normalize power to 1
    power = np.mean(np.abs(signal) ** 2)
    if power > 0:
        signal = signal / np.sqrt(power)
    
    return signal.astype(np.complex64)
