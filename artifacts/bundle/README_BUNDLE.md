# Edge Bundle — Cognitive RF Receiver

## Overview

Self-contained bundle for edge deployment. Includes:
- ONNX model for inference
- Preprocessing configuration
- Policy configuration (calibration + abstention)

## Files

| File | Description |
|------|-------------|
| `model.onnx` | Quantized CNN classifier |
| `preprocess.json` | STFT and normalization parameters |
| `policy.json` | Temperature, thresholds, OOD settings |

## Input Format

- **Type**: Complex IQ samples (complex64)
- **Length**: 4096 samples
- **Sample rate**: 1 MS/s

## Preprocessing Pipeline

1. Load IQ data as complex64
2. Compute STFT (n_fft=256, hop=64, hann window)
3. Take magnitude (dB scale)
4. Normalize per-sample (mean=0, std=1)
5. Reshape to [1, 1, 256, 15]

## Model

- **Input**: float32 tensor [batch, 1, 256, 15]
- **Output**: float32 logits [batch, 5]
- **Classes**: BPSK, QPSK, QAM16, GFSK, NOISE

## Post-Processing (Policy)

1. Apply temperature scaling: `probs = softmax(logits / T)`
2. Get max probability: `conf = max(probs)`
3. Check threshold: `if conf < τ[mode]: label = UNKNOWN`
4. Return prediction with confidence

## Usage

```python
from tools.run_bundle import load_bundle, predict

bundle = load_bundle("artifacts/bundle")
result = bundle.predict(iq_data, mode="TRUSTED")
print(result["label"], result["confidence"])
```

## Latency

Run `python -m tools.run_bundle --benchmark` for:
- p50, p95, p99 latencies
- Throughput (samples/sec)
