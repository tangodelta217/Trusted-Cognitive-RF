# Assurance package for TFM AI
"""
Assurance modules: calibration, OOD detection, abstention.
"""

from .temperature_scaling import TemperatureScaling
from .calibration_metrics import compute_ece, reliability_diagram_data
from .ood_scores import compute_msp, compute_entropy, compute_energy
from .risk_coverage import compute_risk_coverage

__all__ = [
    "TemperatureScaling",
    "compute_ece",
    "reliability_diagram_data",
    "compute_msp",
    "compute_entropy",
    "compute_energy",
    "compute_risk_coverage",
]
