#!/usr/bin/env python3
"""
V0.7 Demo: Reproducible inference demo with calibration and UNKNOWN detection.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.demo.demo_core import run_demo_pipeline
from common.demo.plotting import plot_demo_result
from common.policy.operating_points import load_operating_points, get_preset
from common.infer.onnx_infer import find_model_onnx


def main():
    parser = argparse.ArgumentParser(
        description="V0.7 Demo: RF Signal Classification with UNKNOWN Detection"
    )
    parser.add_argument(
        "--split", type=str, default="test_id",
        choices=["val", "test_id", "test_ood_chan", "test_ood_mod"],
        help="Dataset split"
    )
    parser.add_argument("--index", type=int, default=None, help="Example index")
    parser.add_argument("--random", action="store_true", help="Random index")
    parser.add_argument(
        "--mode", type=str, default="SURVEILLANCE",
        choices=["SURVEILLANCE", "TRUSTED", "CONSERVATIVE"],
        help="Operating mode"
    )
    parser.add_argument("--use_features_cache", action="store_true", default=True)
    parser.add_argument("--no_cache", dest="use_features_cache", action="store_false")
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    # Paths
    dataset_root = project_root / "data" / "datasets" / "v0"
    features_root = project_root / "data" / "features" / "v0"
    
    # Find model
    if args.model_path is None:
        args.model_path = find_model_onnx()
    
    # Load operating points
    op = load_operating_points()
    
    # Get index
    if args.random or args.index is None:
        # Get dataset size
        features_path = features_root / f"{args.split}.npz"
        if features_path.exists():
            data = np.load(features_path)
            n = len(data["y"])
        else:
            data = np.load(dataset_root / f"{args.split}.npz", allow_pickle=True)
            n = len(data["y"])
        args.index = random.randint(0, n - 1)
    
    # Run pipeline
    result = run_demo_pipeline(
        split=args.split,
        index=args.index,
        mode=args.mode,
        dataset_root=dataset_root,
        features_root=features_root,
        model_path=args.model_path,
        operating_points=op,
        use_features_cache=args.use_features_cache,
    )
    
    # Print result
    if not args.quiet:
        print("=" * 60)
        print("V0.7 DEMO — Cognitive RF Receiver")
        print("=" * 60)
        print()
        print(f"Split={result.split}  idx={result.index}  Mode={result.mode}")
        print(f"T={result.temperature:.4f}  τ={result.tau:.4f}")
        print()
        
        status = "UNKNOWN" if result.is_unknown else result.predicted_class
        print(f"Pred={result.predicted_class}  conf={result.confidence:.3f}  → {status}")
        print()
        
        print("Top-3:")
        for p in result.top_k:
            print(f"  {p['class']}: {p['prob']:.3f}")
        print()
        
        print("Latency (ms):")
        print(f"  load:     {result.latency_load_ms:6.2f}")
        print(f"  features: {result.latency_features_ms:6.2f}")
        print(f"  infer:    {result.latency_infer_ms:6.2f}")
        print(f"  policy:   {result.latency_policy_ms:6.2f}")
        print(f"  total:    {result.latency_total_ms:6.2f}")
    
    # Save results
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = project_root / "runs" / "v0_7" / f"demo_{timestamp}"
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plot_path = args.save_dir / "spectrogram.png"
    plot_demo_result(
        features=result.features,
        top_k=result.top_k,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        is_unknown=result.is_unknown,
        mode=result.mode,
        tau=result.tau,
        save_path=plot_path
    )
    
    # Save JSON
    result_path = args.save_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    if not args.quiet:
        print()
        print(f"Saved: {plot_path}")
        print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
