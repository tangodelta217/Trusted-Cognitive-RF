"""
Main dataset generation script for V0 synthetic dataset.

Generates reproducible train/val/test splits with ID and OOD examples.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
from tqdm import tqdm

from .generators import generate_modulated_signal
from .generators.impairments import sample_impairments, apply_impairments
from .io import save_split, get_class_mapping
from .manifest import build_manifest_rows, write_manifest


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load dataset configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_split_sizes(config: Dict[str, Any]) -> Dict[str, int]:
    """
    Get number of examples per class for each split based on preset.
    
    Returns dict with keys: train, val, test_id, test_ood_mod, test_ood_chan
    """
    preset = config["splits"].get("preset", "TINY").lower()
    sizes = config["splits"][preset]
    
    return {
        "train": sizes["train_per_class"],
        "val": sizes["val_per_class"],
        "test_id": sizes["test_id_per_class"],
        "test_ood_mod": sizes["test_ood_mod_per_class"],
        "test_ood_chan": sizes["test_ood_chan_total"],
    }


def generate_split(
    classes: List[str],
    n_per_class: int,
    impairment_config: Dict[str, Any],
    signal_config: Dict[str, Any],
    class_mapping: Dict[str, int],
    base_seed: int,
    split_name: str,
    domain: str,
) -> Tuple[NDArray[np.complex64], NDArray[np.int32], List[str], List[Dict[str, Any]]]:
    """
    Generate a complete split with multiple classes.
    
    Args:
        classes: List of modulation classes to generate.
        n_per_class: Number of examples per class.
        impairment_config: Impairment ranges.
        signal_config: Signal parameters (fs, sps, n_samples, etc.).
        class_mapping: Class name to index mapping.
        base_seed: Base random seed for reproducibility.
        split_name: Name of split for logging.
        domain: Domain type for manifest.
        
    Returns:
        Tuple of (iq_array, labels, label_names, metadata_list).
    """
    n_samples = signal_config["samples_per_example"]
    sps = signal_config["sps"]
    fs_hz = signal_config["fs_hz"]
    rrc_rolloff = signal_config.get("rrc_rolloff", 0.35)
    rrc_span = signal_config.get("rrc_span_symbols", 8)
    
    total = len(classes) * n_per_class
    iq_data = np.zeros((total, n_samples), dtype=np.complex64)
    labels = np.zeros(total, dtype=np.int32)
    label_names = []
    metadata = []
    
    idx = 0
    for cls_idx, cls_name in enumerate(classes):
        class_label = class_mapping[cls_name]
        
        desc = f"{split_name}/{cls_name}"
        for i in tqdm(range(n_per_class), desc=desc, leave=False):
            # Deterministic seed per example
            example_seed = base_seed + idx
            rng = np.random.default_rng(example_seed)
            
            # Generate clean signal
            signal = generate_modulated_signal(
                modulation=cls_name,
                n_samples=n_samples,
                sps=sps,
                rng=rng,
                rrc_rolloff=rrc_rolloff,
                rrc_span_symbols=rrc_span
            )
            
            # Sample and apply impairments
            imp_params = sample_impairments(impairment_config, rng)
            impaired = apply_impairments(signal, imp_params, fs_hz, rng)
            
            # Store
            iq_data[idx] = impaired
            labels[idx] = class_label
            label_names.append(cls_name)
            
            meta = {
                "modulation": cls_name,
                "seed": example_seed,
                **imp_params
            }
            metadata.append(meta)
            
            idx += 1
    
    return iq_data, labels, label_names, metadata


def generate_ood_chan_split(
    classes: List[str],
    n_total: int,
    impairment_config: Dict[str, Any],
    signal_config: Dict[str, Any],
    class_mapping: Dict[str, int],
    base_seed: int,
) -> Tuple[NDArray[np.complex64], NDArray[np.int32], List[str], List[Dict[str, Any]]]:
    """
    Generate OOD-CHAN split: ID classes with harder impairments.
    
    Examples are distributed evenly across ID classes.
    """
    n_samples = signal_config["samples_per_example"]
    sps = signal_config["sps"]
    fs_hz = signal_config["fs_hz"]
    rrc_rolloff = signal_config.get("rrc_rolloff", 0.35)
    rrc_span = signal_config.get("rrc_span_symbols", 8)
    
    iq_data = np.zeros((n_total, n_samples), dtype=np.complex64)
    labels = np.zeros(n_total, dtype=np.int32)
    label_names = []
    metadata = []
    
    n_classes = len(classes)
    
    for idx in tqdm(range(n_total), desc="test_ood_chan", leave=False):
        # Cycle through ID classes
        cls_name = classes[idx % n_classes]
        class_label = class_mapping[cls_name]
        
        # Deterministic seed
        example_seed = base_seed + idx
        rng = np.random.default_rng(example_seed)
        
        # Generate signal
        signal = generate_modulated_signal(
            modulation=cls_name,
            n_samples=n_samples,
            sps=sps,
            rng=rng,
            rrc_rolloff=rrc_rolloff,
            rrc_span_symbols=rrc_span
        )
        
        # Apply harder impairments (OOD channel)
        imp_params = sample_impairments(impairment_config, rng)
        impaired = apply_impairments(signal, imp_params, fs_hz, rng)
        
        iq_data[idx] = impaired
        labels[idx] = class_label
        label_names.append(cls_name)
        
        meta = {
            "modulation": cls_name,
            "seed": example_seed,
            **imp_params
        }
        metadata.append(meta)
    
    return iq_data, labels, label_names, metadata


def make_dataset(
    config_path: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate complete V0 dataset.
    
    Args:
        config_path: Path to dataset config YAML.
        output_dir: Output directory for NPZ and manifest files.
        verbose: If True, print progress info.
        
    Returns:
        Summary dict with generation statistics.
    """
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    base_seed = config["seed"]
    signal_cfg = config["signal"]
    classes_id = config["classes"]["id"]
    classes_ood = config["classes"]["ood_mod"]
    imp_id = config["impairments_id"]
    imp_ood = config["impairments_ood_chan"]
    sizes = get_split_sizes(config)
    
    # Class mapping (ID classes first, then OOD)
    class_mapping = get_class_mapping(classes_id, classes_ood)
    
    if verbose:
        print(f"Dataset: {config['dataset_name']} v{config['version']}")
        print(f"Seed: {base_seed}")
        print(f"ID classes: {classes_id}")
        print(f"OOD-MOD classes: {classes_ood}")
        print(f"Samples per example: {signal_cfg['samples_per_example']}")
        print(f"Preset: {config['splits'].get('preset', 'TINY')}")
        print(f"Sizes: {sizes}")
        print()
    
    all_manifest_rows = []
    summary = {"splits": {}}
    
    # Seed offsets for each split to ensure no overlap
    seed_offsets = {
        "train": 0,
        "val": 100000,
        "test_id": 200000,
        "test_ood_mod": 300000,
        "test_ood_chan": 400000,
    }
    
    # Generate train split (ID classes only)
    if verbose:
        print("Generating train split...")
    iq, y, names, meta = generate_split(
        classes=classes_id,
        n_per_class=sizes["train"],
        impairment_config=imp_id,
        signal_config=signal_cfg,
        class_mapping=class_mapping,
        base_seed=base_seed + seed_offsets["train"],
        split_name="train",
        domain="id",
    )
    save_split(output_dir / "train.npz", iq, y, names, meta,
               extra_info={"config": config, "split": "train"})
    rows = build_manifest_rows("train", "id", meta, "train.npz")
    all_manifest_rows.extend(rows)
    summary["splits"]["train"] = {"total": len(iq), "classes": list(set(names))}
    
    # Generate val split (ID classes only)
    if verbose:
        print("Generating val split...")
    iq, y, names, meta = generate_split(
        classes=classes_id,
        n_per_class=sizes["val"],
        impairment_config=imp_id,
        signal_config=signal_cfg,
        class_mapping=class_mapping,
        base_seed=base_seed + seed_offsets["val"],
        split_name="val",
        domain="id",
    )
    save_split(output_dir / "val.npz", iq, y, names, meta,
               extra_info={"config": config, "split": "val"})
    rows = build_manifest_rows("val", "id", meta, "val.npz")
    all_manifest_rows.extend(rows)
    summary["splits"]["val"] = {"total": len(iq), "classes": list(set(names))}
    
    # Generate test_id split (ID classes, ID impairments)
    if verbose:
        print("Generating test_id split...")
    iq, y, names, meta = generate_split(
        classes=classes_id,
        n_per_class=sizes["test_id"],
        impairment_config=imp_id,
        signal_config=signal_cfg,
        class_mapping=class_mapping,
        base_seed=base_seed + seed_offsets["test_id"],
        split_name="test_id",
        domain="id",
    )
    save_split(output_dir / "test_id.npz", iq, y, names, meta,
               extra_info={"config": config, "split": "test_id"})
    rows = build_manifest_rows("test_id", "id", meta, "test_id.npz")
    all_manifest_rows.extend(rows)
    summary["splits"]["test_id"] = {"total": len(iq), "classes": list(set(names))}
    
    # Generate test_ood_mod split (OOD classes, ID impairments)
    if verbose:
        print("Generating test_ood_mod split...")
    iq, y, names, meta = generate_split(
        classes=classes_ood,
        n_per_class=sizes["test_ood_mod"],
        impairment_config=imp_id,  # Same impairments as ID
        signal_config=signal_cfg,
        class_mapping=class_mapping,
        base_seed=base_seed + seed_offsets["test_ood_mod"],
        split_name="test_ood_mod",
        domain="ood_mod",
    )
    save_split(output_dir / "test_ood_mod.npz", iq, y, names, meta,
               extra_info={"config": config, "split": "test_ood_mod"})
    rows = build_manifest_rows("test_ood_mod", "ood_mod", meta, "test_ood_mod.npz")
    all_manifest_rows.extend(rows)
    summary["splits"]["test_ood_mod"] = {"total": len(iq), "classes": list(set(names))}
    
    # Generate test_ood_chan split (ID classes, OOD impairments)
    if verbose:
        print("Generating test_ood_chan split...")
    iq, y, names, meta = generate_ood_chan_split(
        classes=classes_id,
        n_total=sizes["test_ood_chan"],
        impairment_config=imp_ood,  # Harder impairments
        signal_config=signal_cfg,
        class_mapping=class_mapping,
        base_seed=base_seed + seed_offsets["test_ood_chan"],
    )
    save_split(output_dir / "test_ood_chan.npz", iq, y, names, meta,
               extra_info={"config": config, "split": "test_ood_chan"})
    rows = build_manifest_rows("test_ood_chan", "ood_chan", meta, "test_ood_chan.npz")
    all_manifest_rows.extend(rows)
    summary["splits"]["test_ood_chan"] = {"total": len(iq), "classes": list(set(names))}
    
    # Write manifest
    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, all_manifest_rows)
    
    if verbose:
        print()
        print(f"Dataset saved to: {output_dir}")
        print(f"Manifest: {manifest_path}")
        total = sum(s["total"] for s in summary["splits"].values())
        print(f"Total examples: {total}")
    
    summary["output_dir"] = str(output_dir)
    summary["manifest_path"] = str(manifest_path)
    summary["config"] = config
    
    return summary


if __name__ == "__main__":
    # Quick test
    import sys
    
    config_path = Path(__file__).parent / "configs" / "dataset_v0.yaml"
    output_dir = Path(__file__).parents[2] / "data" / "datasets" / "v0"
    
    make_dataset(config_path, output_dir)
