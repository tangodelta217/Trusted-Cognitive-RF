# Operating Points V0.6

## Concept

The system can operate at different **trust levels** by adjusting the confidence threshold τ:

| Mode | Coverage | Behavior |
|------|----------|----------|
| **SURVEILLANCE** | 95% | Decide on most samples, minimize abstentions |
| **TRUSTED** | 80% | Reject more unknowns, higher accuracy on accepted |
| **CONSERVATIVE** | 70% | Very selective, for critical decisions |

## Trade-off

Lower coverage → Higher τ →
- **Higher accuracy** on accepted samples
- **More OOD rejected** as UNKNOWN
- **Fewer decisions** made

## Usage

```bash
# Run sweep
python scripts/sweep_operating_point_v0_6.py

# Outputs (runs/v0_6/)
# - sweep.csv: metrics per coverage target
# - operating_points.json: preset thresholds
# - plots/: visualizations
```

## Preset Selection

```python
from common.policy import PRESETS, fit_threshold_by_coverage

# Get tau for TRUSTED mode
tau = fit_threshold_by_coverage(conf_val, PRESETS["TRUSTED"]["coverage_target"])
```

## In Production

1. Select operating mode based on mission requirements
2. Apply `tau` from `operating_points.json`
3. Mark samples with `conf < tau` as **UNKNOWN**
