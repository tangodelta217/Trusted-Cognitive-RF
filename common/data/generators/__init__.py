# Generators package
"""
Signal and impairment generators for synthetic IQ data.
"""

from .modulations import generate_modulated_signal, MODULATION_MAP
from .impairments import apply_impairments
from .channel import apply_channel

__all__ = [
    "generate_modulated_signal",
    "MODULATION_MAP",
    "apply_impairments",
    "apply_channel",
]
