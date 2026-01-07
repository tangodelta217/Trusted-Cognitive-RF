# Model V0.4 — CNN Baseline

## Overview

Small CNN for RF signal classification, designed to be quantization-friendly.

## Architecture

```
Input: (B, 1, 256, 15) float32

ConvBlock1: Conv2d(1→8) → BatchNorm → ReLU → MaxPool(2,1)
ConvBlock2: Conv2d(8→16) → BatchNorm → ReLU → MaxPool(2,1)
ConvBlock3: Conv2d(16→32) → BatchNorm → ReLU → MaxPool(2,1)

Flatten → Linear(→128) → ReLU → Dropout(0.2) → Linear(→5)

Output: (B, 5) logits
```

## Specifications

| Parameter | Value |
|-----------|-------|
| Input shape | (1, 256, 15) |
| Num classes | 5 |
| Conv channels | [8, 16, 32] |
| Kernel size | 3×3 |
| Pool | (2, 1) - freq only |
| FC hidden | 128 |
| Dropout | 0.2 |
| Parameters | ~30K |

## Training

```bash
# 1. Build feature cache
python scripts/build_feature_cache_v0.py

# 2. Train
python scripts/train_v0_4.py
```

**Hyperparameters:**
- Epochs: 30 (early stopping patience=7)
- Batch size: 64
- LR: 0.001
- Optimizer: Adam
- Loss: CrossEntropy

## Evaluation

```bash
python scripts/eval_v0_4.py
```

**Splits evaluated:**
- `test_id`: In-distribution (ID classes, ID impairments)
- `test_ood_mod`: OOD modulations (8PSK, 64QAM, CPFSK)
- `test_ood_chan`: OOD channel (ID classes, harder impairments)

## ONNX Export

```bash
# Export
python scripts/export_onnx_v0_4.py

# Verify parity
python scripts/verify_onnx_v0_4.py
```

## Classes

| Index | Class |
|-------|-------|
| 0 | BPSK |
| 1 | QPSK |
| 2 | QAM16 |
| 3 | GFSK |
| 4 | NOISE |

## Quantization Notes

Model uses only quantization-friendly operations:
- Conv2d + BatchNorm (fusable)
- ReLU (trivial quantization)
- MaxPool2d
- Linear

Avoid: Softmax in inference (apply post-quantization), LayerNorm, GELU.
