"""
Generate golden examples for feature extraction verification.

Creates deterministic test signals and their expected feature outputs.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, Any, List
import json
import hashlib


def generate_tone(
    n_samples: int = 2048,
    freq_hz: float = 10000.0,
    fs_hz: float = 1e6,
    seed: int = 42
) -> NDArray[np.complex64]:
    """Generate a pure complex tone."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs_hz
    phase = rng.uniform(0, 2 * np.pi)
    tone = np.exp(1j * (2 * np.pi * freq_hz * t + phase))
    return tone.astype(np.complex64)


def generate_noise(
    n_samples: int = 2048,
    seed: int = 43
) -> NDArray[np.complex64]:
    """Generate complex Gaussian noise."""
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(n_samples) + 
             1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    return noise.astype(np.complex64)


def generate_bpsk_like(
    n_samples: int = 2048,
    sps: int = 8,
    seed: int = 44
) -> NDArray[np.complex64]:
    """Generate a simple BPSK-like signal."""
    rng = np.random.default_rng(seed)
    n_symbols = n_samples // sps + 1
    bits = rng.integers(0, 2, n_symbols)
    symbols = 2 * bits - 1  # {-1, +1}
    signal = np.repeat(symbols, sps)[:n_samples]
    return signal.astype(np.complex64)


def generate_chirp(
    n_samples: int = 2048,
    f0_hz: float = -50000.0,
    f1_hz: float = 50000.0,
    fs_hz: float = 1e6,
    seed: int = 45
) -> NDArray[np.complex64]:
    """Generate a linear chirp signal."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs_hz
    duration = n_samples / fs_hz
    
    # Linear chirp: frequency goes from f0 to f1
    phase = 2 * np.pi * (f0_hz * t + (f1_hz - f0_hz) * t**2 / (2 * duration))
    phase += rng.uniform(0, 2 * np.pi)  # Random initial phase
    
    chirp = np.exp(1j * phase)
    return chirp.astype(np.complex64)


def compute_stable_hash(arr: NDArray[np.float32], decimals: int = 6) -> str:
    """
    Compute a stable hash of a float array.
    
    Rounds to specified decimals before hashing for reproducibility.
    
    Args:
        arr: Float array.
        decimals: Number of decimal places to round.
        
    Returns:
        Hex string hash.
    """
    rounded = np.round(arr, decimals=decimals)
    # Use tobytes for deterministic serialization
    arr_bytes = rounded.astype(np.float32).tobytes()
    return hashlib.sha256(arr_bytes).hexdigest()[:16]


def make_golden_examples(
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate golden examples for feature verification.
    
    Args:
        output_dir: Directory to save golden files.
        verbose: Print progress.
        
    Returns:
        Summary dict.
    """
    # Import here to avoid circular import
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from common.features.extract import extract_features, load_config
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config()
    n_samples = config["input"]["samples_per_example"]
    
    # Define golden signals
    signal_generators = {
        "tone": lambda: generate_tone(n_samples, seed=42),
        "noise": lambda: generate_noise(n_samples, seed=43),
        "bpsk_like": lambda: generate_bpsk_like(n_samples, seed=44),
        "chirp": lambda: generate_chirp(n_samples, seed=45),
    }
    
    inputs = {}
    features = {}
    hashes = {}
    
    for name, generator in signal_generators.items():
        if verbose:
            print(f"Generating golden: {name}")
        
        # Generate input
        iq = generator()
        inputs[name] = iq
        
        # Extract features
        feat = extract_features(iq, config)
        features[name] = feat
        
        # Compute stable hash
        hash_val = compute_stable_hash(feat)
        hashes[name] = hash_val
        
        if verbose:
            print(f"  Shape: {feat.shape}, Hash: {hash_val}")
    
    # Save inputs
    inputs_path = output_dir / "golden_inputs_v0.npz"
    np.savez_compressed(inputs_path, **inputs)
    if verbose:
        print(f"\nSaved inputs to: {inputs_path}")
    
    # Save features
    features_path = output_dir / "golden_features_v0.npz"
    np.savez_compressed(features_path, **features)
    if verbose:
        print(f"Saved features to: {features_path}")
    
    # Save hashes
    hashes_path = output_dir / "golden_hashes_v0.json"
    with open(hashes_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
    if verbose:
        print(f"Saved hashes to: {hashes_path}")
    
    return {
        "inputs_path": str(inputs_path),
        "features_path": str(features_path),
        "hashes_path": str(hashes_path),
        "signals": list(signal_generators.keys()),
        "hashes": hashes,
    }


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    make_golden_examples(output_dir)
