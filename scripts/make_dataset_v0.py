#!/usr/bin/env python3
"""
CLI script for generating V0 synthetic RF dataset.

Usage:
    python scripts/make_dataset_v0.py --config common/data/configs/dataset_v0.yaml --out data/datasets/v0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.data.make_dataset_v0 import make_dataset, load_config
from common.data.io import summarize_split
from common.data.manifest import summarize_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate V0 synthetic RF dataset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "common" / "data" / "configs" / "dataset_v0.yaml",
        help="Path to dataset config YAML"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "data" / "datasets" / "v0",
        help="Output directory for dataset files"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification checks after generation"
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    print("=" * 60)
    print("V0 Synthetic RF Dataset Generator")
    print("=" * 60)
    print()
    
    summary = make_dataset(
        config_path=args.config,
        output_dir=args.out,
        verbose=not args.quiet
    )
    
    if args.verify:
        print()
        print("=" * 60)
        print("Verification")
        print("=" * 60)
        
        # Verify each split
        for split_name in ["train", "val", "test_id", "test_ood_mod", "test_ood_chan"]:
            filepath = args.out / f"{split_name}.npz"
            if filepath.exists():
                info = summarize_split(filepath)
                print(f"\n{split_name}:")
                print(f"  Examples: {info['n_examples']}")
                print(f"  Samples per example: {info['n_samples']}")
                print(f"  Dtype: {info['dtype']}")
                print(f"  SNR range: {info['snr_range']}")
                print(f"  Classes: {info['class_distribution']}")
        
        # Verify manifest
        manifest_path = args.out / "manifest.csv"
        if manifest_path.exists():
            manifest_summary = summarize_manifest(manifest_path)
            print(f"\nManifest total: {manifest_summary['total']}")
            print(f"By split: {manifest_summary['by_split']}")
            print(f"By domain: {manifest_summary['by_domain']}")
        
        # Verify OOD separation
        print("\n--- OOD Separation Check ---")
        config = load_config(args.config)
        id_classes = set(config["classes"]["id"])
        ood_classes = set(config["classes"]["ood_mod"])
        
        for split in ["train", "val"]:
            split_info = summarize_split(args.out / f"{split}.npz")
            split_classes = set(split_info["class_distribution"].keys())
            ood_in_split = split_classes & ood_classes
            if ood_in_split:
                print(f"  ERROR: OOD classes found in {split}: {ood_in_split}")
            else:
                print(f"  OK: {split} contains only ID classes")
        
        print()
        print("Verification complete.")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
