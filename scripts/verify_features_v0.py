#!/usr/bin/env python3
"""
Verify feature extraction against golden examples.

Checks:
1. Features match golden features (allclose)
2. Hashes match expected hashes

Usage:
    python scripts/verify_features_v0.py
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.features.extract import extract_features, load_config
from common.features.golden.make_golden_v0 import compute_stable_hash


def verify_features(
    golden_dir: Path,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    verbose: bool = True
) -> bool:
    """
    Verify feature extraction against golden examples.
    
    Args:
        golden_dir: Path to golden files directory.
        rtol: Relative tolerance for allclose.
        atol: Absolute tolerance for allclose.
        verbose: Print progress.
        
    Returns:
        True if all checks pass.
    """
    golden_dir = Path(golden_dir)
    
    # Check files exist
    inputs_path = golden_dir / "golden_inputs_v0.npz"
    features_path = golden_dir / "golden_features_v0.npz"
    hashes_path = golden_dir / "golden_hashes_v0.json"
    
    for path in [inputs_path, features_path, hashes_path]:
        if not path.exists():
            print(f"ERROR: Missing golden file: {path}")
            print("Run 'python scripts/make_golden_v0.py' first.")
            return False
    
    # Load golden data
    golden_inputs = np.load(inputs_path)
    golden_features = np.load(features_path)
    with open(hashes_path, "r") as f:
        golden_hashes = json.load(f)
    
    config = load_config()
    
    all_passed = True
    
    for name in golden_inputs.files:
        if verbose:
            print(f"Verifying: {name}")
        
        # Extract features from input
        iq = golden_inputs[name]
        features = extract_features(iq, config)
        
        # Check 1: Shape and dtype
        expected_shape = golden_features[name].shape
        if features.shape != expected_shape:
            print(f"  FAIL: Shape mismatch: {features.shape} vs {expected_shape}")
            all_passed = False
            continue
        
        if features.dtype != np.float32:
            print(f"  FAIL: Dtype mismatch: {features.dtype} vs float32")
            all_passed = False
            continue
        
        # Check 2: No NaNs or Infs
        if np.any(np.isnan(features)):
            print(f"  FAIL: NaN detected")
            all_passed = False
            continue
        
        if np.any(np.isinf(features)):
            print(f"  FAIL: Inf detected")
            all_passed = False
            continue
        
        # Check 3: Values match (allclose)
        expected_features = golden_features[name]
        if not np.allclose(features, expected_features, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(features - expected_features))
            print(f"  FAIL: Values differ (max diff: {max_diff:.2e})")
            all_passed = False
            continue
        
        # Check 4: Hash matches
        current_hash = compute_stable_hash(features)
        expected_hash = golden_hashes[name]
        if current_hash != expected_hash:
            print(f"  FAIL: Hash mismatch: {current_hash} vs {expected_hash}")
            all_passed = False
            continue
        
        if verbose:
            print(f"  OK: Shape={features.shape}, Hash={current_hash}")
    
    return all_passed


def main():
    print("=" * 60)
    print("Feature Extraction Verification (V0)")
    print("=" * 60)
    print()
    
    golden_dir = project_root / "common" / "features" / "golden"
    
    passed = verify_features(golden_dir, verbose=True)
    
    print()
    print("=" * 60)
    if passed:
        print("RESULT: ALL CHECKS PASSED ✓")
    else:
        print("RESULT: SOME CHECKS FAILED ✗")
    print("=" * 60)
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
