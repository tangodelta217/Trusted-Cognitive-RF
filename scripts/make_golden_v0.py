#!/usr/bin/env python3
"""
CLI script for generating golden feature examples.

Usage:
    python scripts/make_golden_v0.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.features.golden.make_golden_v0 import make_golden_examples


def main():
    print("=" * 60)
    print("Golden Feature Examples Generator (V0)")
    print("=" * 60)
    print()
    
    # Output to golden directory
    output_dir = project_root / "common" / "features" / "golden"
    
    summary = make_golden_examples(output_dir, verbose=True)
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Signals: {summary['signals']}")
    print(f"Hashes: {summary['hashes']}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
