"""
STFT (Short-Time Fourier Transform) implementation.

Provides deterministic STFT computation for IQ signals.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Optional


def get_window(window_type: str, length: int) -> NDArray[np.float64]:
    """
    Get window function by name.
    
    Args:
        window_type: Window type ('hann', 'hamming', 'blackman', 'rect').
        length: Window length in samples.
        
    Returns:
        Window coefficients.
    """
    window_type = window_type.lower()
    
    if window_type == "hann":
        return np.hanning(length)
    elif window_type == "hamming":
        return np.hamming(length)
    elif window_type == "blackman":
        return np.blackman(length)
    elif window_type in ("rect", "rectangular", "boxcar"):
        return np.ones(length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def compute_stft(
    iq: NDArray[np.complex64],
    n_fft: int = 256,
    hop: int = 128,
    window: Optional[NDArray[np.float64]] = None,
    fftshift: bool = True
) -> NDArray[np.complex64]:
    """
    Compute Short-Time Fourier Transform.
    
    Args:
        iq: Input complex IQ signal, shape (N,).
        n_fft: FFT size.
        hop: Hop size between frames.
        window: Window function. If None, uses Hann window.
        fftshift: If True, shift DC to center.
        
    Returns:
        STFT output, shape (n_fft, n_frames) complex64.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n_samples = len(iq)
    
    # Default window
    if window is None:
        window = get_window("hann", n_fft)
    window = np.asarray(window, dtype=np.float64)
    
    # Calculate number of frames
    n_frames = (n_samples - n_fft) // hop + 1
    
    if n_frames < 1:
        raise ValueError(
            f"Signal too short ({n_samples}) for STFT with n_fft={n_fft}"
        )
    
    # Allocate output
    stft_out = np.zeros((n_fft, n_frames), dtype=np.complex64)
    
    # Compute STFT frame by frame
    for i in range(n_frames):
        start = i * hop
        end = start + n_fft
        frame = iq[start:end] * window
        spectrum = np.fft.fft(frame)
        
        if fftshift:
            spectrum = np.fft.fftshift(spectrum)
        
        stft_out[:, i] = spectrum.astype(np.complex64)
    
    return stft_out


def stft_from_config(
    iq: NDArray[np.complex64],
    config: Dict[str, Any]
) -> NDArray[np.complex64]:
    """
    Compute STFT using configuration dict.
    
    Args:
        iq: Input IQ signal.
        config: STFT config dict with keys:
            - n_fft: int
            - hop: int
            - window: str
            - fftshift: bool
            
    Returns:
        STFT output, shape (n_fft, n_frames).
    """
    n_fft = config.get("n_fft", 256)
    hop = config.get("hop", 128)
    window_type = config.get("window", "hann")
    do_fftshift = config.get("fftshift", True)
    
    window = get_window(window_type, n_fft)
    
    return compute_stft(
        iq=iq,
        n_fft=n_fft,
        hop=hop,
        window=window,
        fftshift=do_fftshift
    )
