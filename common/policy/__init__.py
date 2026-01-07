# Policy package
"""
Abstention and operating point policies.
"""

from .abstention import (
    fit_threshold_by_coverage,
    apply_abstention,
    PRESETS,
)

__all__ = [
    "fit_threshold_by_coverage",
    "apply_abstention",
    "PRESETS",
]
