#!/usr/bin/env python3
"""
CLI wrapper for verifying ONNX model parity.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.export.verify_onnx import verify_from_run


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in: {runs_dir}")
    
    return runs[-1]


def main():
    parser = argparse.ArgumentParser(description="Verify ONNX model parity")
    parser.add_argument(
        "--run_dir", type=Path, default=None,
        help="Run directory (default: latest in runs/v0_4)"
    )
    parser.add_argument(
        "--features_root", type=Path,
        default=project_root / "data" / "features" / "v0",
        help="Path to cached features"
    )
    parser.add_argument(
        "--split", type=str, default="test_id",
        help="Split to use for verification"
    )
    parser.add_argument(
        "--n_examples", type=int, default=100,
        help="Number of examples to test"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ONNX Parity Verification")
    print("=" * 60)
    print()
    
    # Find run directory
    if args.run_dir is None:
        args.run_dir = find_latest_run(project_root / "runs" / "v0_4")
    
    print(f"Run directory: {args.run_dir}")
    print(f"Features root: {args.features_root}")
    print(f"Split: {args.split}")
    print(f"Examples: {args.n_examples}")
    print()
    
    result = verify_from_run(
        run_dir=args.run_dir,
        features_root=args.features_root,
        split=args.split,
        n_examples=args.n_examples,
        verbose=True
    )
    
    print()
    print("=" * 60)
    if result["passed"]:
        print("VERIFICATION PASSED ✓")
    else:
        print("VERIFICATION FAILED ✗")
    print("=" * 60)
    
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
