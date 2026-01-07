"""
Manifest generation for dataset tracking.

Creates CSV files with per-example metadata for traceability.
"""

import csv
from pathlib import Path
from typing import Dict, Any, List


def build_manifest_rows(
    split_name: str,
    domain: str,
    examples_metadata: List[Dict[str, Any]],
    file_basename: str,
) -> List[Dict[str, Any]]:
    """
    Build manifest rows for a set of examples.
    
    Args:
        split_name: Name of the split (train, val, test_id, etc.).
        domain: Domain type (id, ood_mod, ood_chan).
        examples_metadata: List of per-example metadata dicts.
        file_basename: NPZ file basename (without path).
        
    Returns:
        List of row dicts for CSV.
    """
    rows = []
    for idx, meta in enumerate(examples_metadata):
        row = {
            "split": split_name,
            "domain": domain,
            "index": idx,
            "file": file_basename,
            "label": meta.get("modulation", ""),
            "snr_db": meta.get("snr_db", ""),
            "cfo_hz": meta.get("cfo_hz", ""),
            "gain": meta.get("gain", ""),
            "phase_rad": meta.get("phase_rad", ""),
            "multipath_taps": meta.get("multipath_taps", 0),
            "seed": meta.get("seed", ""),
        }
        rows.append(row)
    return rows


def write_manifest(
    filepath: Path,
    rows: List[Dict[str, Any]]
) -> None:
    """
    Write manifest rows to CSV file.
    
    Args:
        filepath: Output CSV path.
        rows: List of row dicts.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if not rows:
        return
    
    fieldnames = [
        "split", "domain", "index", "file", "label",
        "snr_db", "cfo_hz", "gain", "phase_rad", "multipath_taps", "seed"
    ]
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_manifest(
    filepath: Path,
    rows: List[Dict[str, Any]]
) -> None:
    """
    Append rows to existing manifest or create new.
    
    Args:
        filepath: CSV file path.
        rows: Rows to append.
    """
    filepath = Path(filepath)
    file_exists = filepath.exists()
    
    fieldnames = [
        "split", "domain", "index", "file", "label",
        "snr_db", "cfo_hz", "gain", "phase_rad", "multipath_taps", "seed"
    ]
    
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def read_manifest(filepath: Path) -> List[Dict[str, Any]]:
    """
    Read manifest from CSV file.
    
    Args:
        filepath: CSV file path.
        
    Returns:
        List of row dicts.
    """
    filepath = Path(filepath)
    rows = []
    
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["index"] = int(row["index"])
            row["snr_db"] = float(row["snr_db"]) if row["snr_db"] else None
            row["cfo_hz"] = float(row["cfo_hz"]) if row["cfo_hz"] else None
            row["gain"] = float(row["gain"]) if row["gain"] else None
            row["phase_rad"] = float(row["phase_rad"]) if row["phase_rad"] else None
            row["multipath_taps"] = int(row["multipath_taps"]) if row["multipath_taps"] else 0
            rows.append(row)
    
    return rows


def summarize_manifest(filepath: Path) -> Dict[str, Any]:
    """
    Generate summary statistics from manifest.
    
    Args:
        filepath: Manifest CSV path.
        
    Returns:
        Summary dict with counts by split, domain, label.
    """
    rows = read_manifest(filepath)
    
    # Count by split
    splits = {}
    for row in rows:
        split = row["split"]
        splits[split] = splits.get(split, 0) + 1
    
    # Count by domain
    domains = {}
    for row in rows:
        domain = row["domain"]
        domains[domain] = domains.get(domain, 0) + 1
    
    # Count by label within each split
    split_labels = {}
    for row in rows:
        split = row["split"]
        label = row["label"]
        if split not in split_labels:
            split_labels[split] = {}
        split_labels[split][label] = split_labels[split].get(label, 0) + 1
    
    return {
        "total": len(rows),
        "by_split": splits,
        "by_domain": domains,
        "by_split_label": split_labels,
    }
