#!/usr/bin/env python3
"""
Prediction CLI with UNKNOWN detection based on operating mode.

Usage:
    python -m tools.predict --mode trusted --input demo_samples/ood.iq
    python -m tools.predict --mode surveillance --split test_id --index 5
    python -m tools.predict --help
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Class names
CLASS_NAMES = ["BPSK", "QPSK", "QAM16", "GFSK", "NOISE"]


def load_policy() -> Dict[str, Any]:
    """Load temperature and thresholds from artifacts."""
    policy = {}
    
    # Temperature
    temp_path = project_root / "artifacts" / "policy" / "temperature.json"
    if temp_path.exists():
        with open(temp_path) as f:
            policy["temperature"] = json.load(f)["temperature"]
    else:
        policy["temperature"] = 1.0
    
    # Thresholds
    thresh_path = project_root / "artifacts" / "policy" / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            data = json.load(f)
            policy["thresholds"] = {m: info["tau"] for m, info in data["modes"].items()}
    else:
        # Fallback defaults
        policy["thresholds"] = {
            "SURVEILLANCE": 0.48,
            "TRUSTED": 0.60,
            "CONSERVATIVE": 0.68,
        }
    
    return policy


def load_sample_from_split(split: str, index: int) -> np.ndarray:
    """Load a sample from a dataset split."""
    features_path = project_root / "data" / "features" / "v0" / f"{split}.npz"
    if not features_path.exists():
        # Try raw data
        raw_path = project_root / "data" / "datasets" / "v0" / f"{split}.npz"
        if raw_path.exists():
            data = np.load(raw_path, allow_pickle=True)
            return data["x"][index]
    else:
        data = np.load(features_path)
        return data["X"][index]
    
    raise FileNotFoundError(f"Could not load split: {split}")


def load_sample_from_file(path: Path) -> np.ndarray:
    """Load a sample from a file."""
    if path.suffix == ".npz":
        data = np.load(path)
        if "x" in data:
            return data["x"]
        elif "X" in data:
            return data["X"]
    elif path.suffix in [".iq", ".raw"]:
        # Load raw IQ as complex64
        return np.fromfile(path, dtype=np.complex64)
    elif path.suffix == ".npy":
        return np.load(path)
    
    raise ValueError(f"Unsupported file format: {path.suffix}")


def predict_with_features(features: np.ndarray, policy: Dict, mode: str) -> Dict[str, Any]:
    """Run prediction on pre-extracted features."""
    from tfm_ai_utamed.assurance.temperature_scaling import softmax
    
    # Load model
    model_path = project_root / "runs" / "v0_4"
    model_dirs = sorted([d for d in model_path.iterdir() if d.is_dir()]) if model_path.exists() else []
    
    if not model_dirs:
        raise FileNotFoundError("No model found. Run training first.")
    
    onnx_path = model_dirs[-1] / "model.onnx"
    
    # Load ONNX model
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        
        # Ensure correct shape
        if features.ndim == 3:
            features = features[np.newaxis, ...]
        
        # Run inference
        logits = session.run(None, {input_name: features.astype(np.float32)})[0]
    except ImportError:
        # Fallback to PyTorch
        import torch
        from common.models.baseline_cnn import BaselineCNN
        
        pt_path = model_dirs[-1] / "model.pt"
        model = BaselineCNN(n_classes=5)
        model.load_state_dict(torch.load(pt_path, map_location='cpu', weights_only=True))
        model.eval()
        
        if features.ndim == 3:
            features = features[np.newaxis, ...]
        
        with torch.no_grad():
            logits = model(torch.from_numpy(features.astype(np.float32))).numpy()
    
    # Apply temperature scaling
    T = policy["temperature"]
    probs = softmax(logits.astype(np.float32), T)
    
    # Get prediction
    pred_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])
    
    # Get threshold for mode
    mode_upper = mode.upper()
    if mode_upper not in policy["thresholds"]:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(policy['thresholds'].keys())}")
    
    tau = policy["thresholds"][mode_upper]
    is_unknown = confidence < tau
    
    # Top-3 predictions
    top_k_idx = np.argsort(probs[0])[::-1][:3]
    top_k = [
        {"class": CLASS_NAMES[i], "prob": float(probs[0, i])}
        for i in top_k_idx
    ]
    
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_idx": pred_idx,
        "confidence": confidence,
        "mode": mode_upper,
        "threshold": tau,
        "is_unknown": is_unknown,
        "label": "UNKNOWN" if is_unknown else CLASS_NAMES[pred_idx],
        "top_k": top_k,
        "temperature": T,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prediction CLI with UNKNOWN detection"
    )
    parser.add_argument("--mode", type=str, default="TRUSTED",
                        choices=["SURVEILLANCE", "TRUSTED", "CONSERVATIVE",
                                 "surveillance", "trusted", "conservative"],
                        help="Operating mode")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input file (.npz, .iq, .npy)")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test_id", "test_ood_mod", "test_ood_chan"],
                        help="Dataset split to use")
    parser.add_argument("--index", type=int, default=0,
                        help="Sample index (with --split)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    
    args = parser.parse_args()
    
    # Validate args
    if args.input is None and args.split is None:
        parser.error("Either --input or --split must be specified")
    
    # Load policy
    policy = load_policy()
    
    # Load features
    if args.input:
        # Load from file
        sample = load_sample_from_file(args.input)
        
        # Extract features if needed
        if sample.dtype == np.complex64:
            from common.features.extract import extract_features, load_config
            config = load_config()
            features = extract_features(sample, config)
        else:
            features = sample
    else:
        # Load from split
        features = load_sample_from_split(args.split, args.index)
    
    # Run prediction
    result = predict_with_features(features, policy, args.mode)
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 50)
        print("Prediction Result")
        print("=" * 50)
        print(f"Mode: {result['mode']}")
        print(f"Threshold (τ): {result['threshold']:.4f}")
        print(f"Temperature (T): {result['temperature']:.4f}")
        print()
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print()
        
        if result['is_unknown']:
            print(f">>> LABEL: UNKNOWN (conf {result['confidence']:.4f} < τ {result['threshold']:.4f})")
        else:
            print(f">>> LABEL: {result['predicted_class']}")
        
        print()
        print("Top-3 predictions:")
        for p in result['top_k']:
            marker = " *" if p['class'] == result['predicted_class'] else ""
            print(f"  {p['class']}: {p['prob']:.4f}{marker}")


if __name__ == "__main__":
    main()
