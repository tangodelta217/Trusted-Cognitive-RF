"""
IO utilities for dataset files.

Handles saving/loading of NPZ files with IQ data and metadata.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, Any, List, Optional
import json


def save_split(
    filepath: Path,
    iq: NDArray[np.complex64],
    labels: NDArray[np.int32],
    label_names: List[str],
    metadata: List[Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a dataset split to NPZ format.
    
    Args:
        filepath: Output NPZ file path.
        iq: Complex IQ data, shape (N, n_samples).
        labels: Integer labels, shape (N,).
        label_names: List of string labels parallel to labels array.
        metadata: List of dicts with per-example metadata.
        extra_info: Optional dict with global info (seed, config, etc.).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract metadata arrays
    snr_db = np.array([m.get("snr_db", np.nan) for m in metadata], dtype=np.float32)
    cfo_hz = np.array([m.get("cfo_hz", np.nan) for m in metadata], dtype=np.float32)
    gain = np.array([m.get("gain", np.nan) for m in metadata], dtype=np.float32)
    phase_rad = np.array([m.get("phase_rad", np.nan) for m in metadata], dtype=np.float32)
    multipath_taps = np.array([m.get("multipath_taps", 0) for m in metadata], dtype=np.int32)
    
    # Build save dict
    save_dict = {
        "iq": iq.astype(np.complex64),
        "y": labels.astype(np.int32),
        "label_names": np.array(label_names, dtype=str),
        "snr_db": snr_db,
        "cfo_hz": cfo_hz,
        "gain": gain,
        "phase_rad": phase_rad,
        "multipath_taps": multipath_taps,
    }
    
    # Add extra info as JSON string
    if extra_info:
        save_dict["info"] = np.array([json.dumps(extra_info)], dtype=str)
    
    np.savez_compressed(filepath, **save_dict)


def load_split(filepath: Path) -> Dict[str, Any]:
    """
    Load a dataset split from NPZ format.
    
    Args:
        filepath: Input NPZ file path.
        
    Returns:
        Dict with keys: iq, y, label_names, snr_db, cfo_hz, etc.
    """
    filepath = Path(filepath)
    data = np.load(filepath, allow_pickle=True)
    
    result = {
        "iq": data["iq"],
        "y": data["y"],
        "label_names": data["label_names"],
        "snr_db": data["snr_db"],
        "cfo_hz": data["cfo_hz"],
        "gain": data.get("gain"),
        "phase_rad": data.get("phase_rad"),
        "multipath_taps": data.get("multipath_taps"),
    }
    
    # Parse extra info if present
    if "info" in data:
        result["info"] = json.loads(str(data["info"][0]))
    
    return result


def get_class_mapping(classes_id: List[str], classes_ood: List[str]) -> Dict[str, int]:
    """
    Create a consistent class name to index mapping.
    
    ID classes come first, then OOD classes.
    
    Args:
        classes_id: List of ID class names.
        classes_ood: List of OOD class names.
        
    Returns:
        Dict mapping class name -> index.
    """
    all_classes = classes_id + classes_ood
    return {name: idx for idx, name in enumerate(all_classes)}


def summarize_split(filepath: Path) -> Dict[str, Any]:
    """
    Print summary statistics for a dataset split.
    
    Args:
        filepath: Path to NPZ file.
        
    Returns:
        Summary dict with counts and statistics.
    """
    data = load_split(filepath)
    
    iq = data["iq"]
    y = data["y"]
    labels = data["label_names"]
    snr = data["snr_db"]
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {labels[i]: int(counts[list(unique).index(i)]) 
                  for i in unique if i < len(labels)}
    
    summary = {
        "filepath": str(filepath),
        "n_examples": len(iq),
        "n_samples": iq.shape[1] if len(iq.shape) > 1 else 0,
        "dtype": str(iq.dtype),
        "class_distribution": class_dist,
        "snr_range": [float(np.nanmin(snr)), float(np.nanmax(snr))],
    }
    
    return summary
