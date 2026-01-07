# Demo V0.7

Reproducible demo showing RF classification with calibration and UNKNOWN detection.

## Usage

```bash
# Basic (test_id, SURVEILLANCE mode)
python scripts/demo_v0_7.py

# Specific example
python scripts/demo_v0_7.py --split test_id --index 42 --mode SURVEILLANCE

# Random OOD example with TRUSTED mode
python scripts/demo_v0_7.py --split test_ood_mod --random --mode TRUSTED

# Channel shift with CONSERVATIVE mode
python scripts/demo_v0_7.py --split test_ood_chan --index 0 --mode CONSERVATIVE
```

## Operating Modes

| Mode | τ | Behavior |
|------|---|----------|
| SURVEILLANCE | 0.4825 | Decide often, few UNKNOWNs |
| TRUSTED | 0.5981 | Balance accuracy/coverage |
| CONSERVATIVE | 0.6786 | Very selective, many UNKNOWNs |

## Output

- `runs/v0_7/demo_<timestamp>/spectrogram.png`
- `runs/v0_7/demo_<timestamp>/result.json`

## Example Console Output

```
Split=test_ood_mod  idx=0  Mode=TRUSTED
T=1.7358  τ=0.5981

Pred=QPSK  conf=0.432  → UNKNOWN

Top-3:
  QPSK: 0.432
  QAM16: 0.311
  BPSK: 0.124

Latency (ms):
  load:     0.45
  features: 0.00
  infer:    1.23
  policy:   0.05
  total:    1.73
```
