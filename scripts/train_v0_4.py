#!/usr/bin/env python3
"""
CLI wrapper for training V0.4 baseline model.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.train.train_v0_4 import train_model, load_train_config
from common.models.cnn_small import load_model_config


def main():
    parser = argparse.ArgumentParser(description="Train V0.4 baseline model")
    parser.add_argument(
        "--train_config", type=Path,
        default=project_root / "common" / "train" / "configs" / "train_v0_4.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--model_config", type=Path,
        default=project_root / "common" / "models" / "configs" / "model_cnn_small_v0.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Run name (default: timestamp)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V0.4 Baseline Training")
    print("=" * 60)
    print()
    
    train_cfg = load_train_config(args.train_config)
    model_cfg = load_model_config(args.model_config)
    
    # Create run directory
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / train_cfg["run"]["out_dir"] / f"run_{run_name}"
    
    print(f"Train config: {args.train_config}")
    print(f"Model config: {args.model_config}")
    print(f"Run directory: {run_dir}")
    print()
    
    summary = train_model(train_cfg, model_cfg, run_dir, verbose=True)
    
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best val accuracy: {summary['best_val_acc']:.4f}")
    print(f"Run directory: {summary['run_dir']}")


if __name__ == "__main__":
    main()
