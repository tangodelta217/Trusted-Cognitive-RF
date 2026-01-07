# Assurance V0.5 — Calibration + UNKNOWN + OOD Detection

## Overview

Post-training assurance for the baseline model: calibration, abstention, OOD metrics.

## Components

### Temperature Scaling
Learns `T > 0` to minimize NLL on validation set.
- Before: raw logits → softmax
- After: logits/T → softmax (better calibrated)

### UNKNOWN Abstention
Threshold τ based on target coverage (e.g., 95%):
- Accept prediction if max_prob ≥ τ
- Otherwise → UNKNOWN

### OOD Detection Scores
| Score | Formula | Higher = |
|-------|---------|----------|
| MSP | max(softmax) | More confident (ID) |
| Entropy | -Σ p log p | More uncertain (OOD) |
| Energy | -T log Σexp(z/T) | More OOD |

## Metrics

### Calibration
- **ECE** (Expected Calibration Error): Lower is better
- **Reliability diagram**: Accuracy vs confidence by bin

### Selective Prediction
- **Risk-coverage curve**: Error rate vs fraction accepted
- **Accuracy on accepted**: When τ applied

### OOD Detection
- **AUROC**: ID vs OOD discrimination
- **AUPR**: Precision-recall balance
- **OOD abstention rate**: % of OOD rejected by τ

## Pipeline

```bash
# 1. Collect logits
python scripts/collect_logits_v0_5.py

# 2. Fit temperature
python scripts/fit_temperature_v0_5.py

# 3. Evaluate
python scripts/eval_assurance_v0_5.py

# 4. Generate plots
python scripts/plot_assurance_v0_5.py
```

## Outputs (runs/v0_5/)
- `temperature.json` - Fitted T
- `threshold.json` - τ for coverage
- `assurance_report.json` - All metrics
- `plots/` - Visualizations
