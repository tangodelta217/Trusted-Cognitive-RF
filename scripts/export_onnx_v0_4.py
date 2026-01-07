#!/usr/bin/env python3
"""
CLI wrapper for exporting V0.4 model to ONNX.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.export.export_onnx import export_from_checkpoint


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in: {runs_dir}")
    
    return runs[-1]


def main():
    parser = argparse.ArgumentParser(description="Export V0.4 model to ONNX")
    parser.add_argument(
        "--run_dir", type=Path, default=None,
        help="Run directory (default: latest in runs/v0_4)"
    )
    parser.add_argument(
        "--model_config", type=Path,
        default=project_root / "common" / "models" / "configs" / "model_cnn_small_v0.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--output", type=str, default="model.onnx",
        help="Output filename (relative to run_dir)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V0.4 ONNX Export")
    print("=" * 60)
    print()
    
    # Find run directory
    if args.run_dir is None:
        args.run_dir = find_latest_run(project_root / "runs" / "v0_4")
    
    checkpoint_path = args.run_dir / "best_model.pt"
    output_path = args.run_dir / args.output
    
    print(f"Run directory: {args.run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print()
    
    export_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_config_path=args.model_config,
        output_path=output_path,
        verbose=True
    )
    
    print()
    print("Export complete!")


if __name__ == "__main__":
    main()
