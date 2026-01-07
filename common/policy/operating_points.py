"""
Operating points loader and selector.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


# Default operating points (from V0.6)
DEFAULT_OPERATING_POINTS = {
    "temperature": 1.7358490974294032,
    "presets": {
        "SURVEILLANCE": {
            "coverage_target": 0.95,
            "tau": 0.48250147700309753,
            "description": "High coverage, minimize abstentions. For monitoring mode.",
        },
        "TRUSTED": {
            "coverage_target": 0.80,
            "tau": 0.5981497764587402,
            "description": "Higher trust, reject more unknowns. For critical decisions.",
        },
        "CONSERVATIVE": {
            "coverage_target": 0.70,
            "tau": 0.6786162853240967,
            "description": "Very selective, high accuracy on accepted. For high-stakes.",
        },
    }
}


def load_operating_points(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load operating points from JSON file.
    
    Args:
        path: Path to operating_points.json. If None, uses defaults.
        
    Returns:
        Operating points dict.
    """
    if path is None:
        # Try to find in runs/v0_6
        project_root = Path(__file__).parents[2]
        runs_v0_6 = project_root / "runs" / "v0_6"
        
        if runs_v0_6.exists():
            runs = sorted([r for r in runs_v0_6.iterdir() if r.is_dir()])
            if runs:
                path = runs[-1] / "operating_points.json"
    
    if path is not None and Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    
    return DEFAULT_OPERATING_POINTS


def get_preset(
    operating_points: Dict[str, Any],
    mode: str = "SURVEILLANCE"
) -> Dict[str, Any]:
    """
    Get preset configuration.
    
    Args:
        operating_points: Operating points dict.
        mode: Preset name (SURVEILLANCE, TRUSTED, CONSERVATIVE).
        
    Returns:
        Dict with tau, coverage_target, description.
    """
    mode = mode.upper()
    presets = operating_points.get("presets", DEFAULT_OPERATING_POINTS["presets"])
    
    if mode not in presets:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(presets.keys())}")
    
    return presets[mode]


def get_temperature(operating_points: Dict[str, Any]) -> float:
    """Get temperature from operating points."""
    return operating_points.get("temperature", DEFAULT_OPERATING_POINTS["temperature"])


# Class names (ID classes)
CLASS_NAMES = ["BPSK", "QPSK", "QAM16", "GFSK", "NOISE"]
