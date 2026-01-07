# Feature Extraction Pipeline V0

## Overview

Deterministic STFT-based feature extraction for RF ML models.
Converts IQ samples (2048 complex) to spectrograms (1, 256, 15) float32.

## Pipeline

```
IQ (2048 complex64)
       ↓
  DC Removal (iq - mean)
       ↓
  RMS Normalize (iq / rms)
       ↓
  STFT (n_fft=256, hop=128, Hann, fftshift)
       ↓
  Power (|X|²)
       ↓
  Log Compress (log1p)
       ↓
  Standardize ((S - mean) / std)
       ↓
Features (1, 256, 15) float32
```

## Output Shape

| Dimension | Value | Description |
|-----------|-------|-------------|
| C | 1 | Channels |
| F | 256 | Frequency bins (n_fft) |
| T | 15 | Time frames |

## Configuration

File: `common/features/configs/features_v0.yaml`

| Parameter | Value |
|-----------|-------|
| n_fft | 256 |
| hop | 128 |
| window | Hann |
| fftshift | true |
| normalize | per_example_standardize |

## Usage

```python
from common.features import extract_features, load_config

# Load config
config = load_config()

# Extract from single IQ
features = extract_features(iq, config)  # (1, 256, 15)

# Extract from batch
features_batch = extract_batch(iq_batch, config)  # (B, 1, 256, 15)
```

## Verification

```bash
# Generate golden examples
python scripts/make_golden_v0.py

# Verify against golden
python scripts/verify_features_v0.py

# Run unit tests
python -m unittest common.features.tests.test_features_v0
```

## Golden Examples

| Signal | Description |
|--------|-------------|
| tone | Pure complex tone |
| noise | AWGN |
| bpsk_like | Simple BPSK |
| chirp | Linear frequency sweep |

## Rationale

- **RMS normalize**: Removes amplitude as a variable, improves robustness
- **fftshift**: Centers DC for visual/statistical consistency
- **log1p**: Numerically stable, compresses dynamic range
- **Per-example norm**: Simple, reproducible, no global stats needed
