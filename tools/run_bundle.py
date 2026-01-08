#!/usr/bin/env python3
"""
Run Edge Bundle: Load and execute the packaged ONNX model with policy.

Usage:
    python -m tools.run_bundle                  # Quick test
    python -m tools.run_bundle --benchmark      # Latency benchmark (200 runs)
    python -m tools.run_bundle --input sample.npz --mode conservative
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class EdgeBundle:
    """Self-contained edge inference bundle."""
    
    def __init__(self, bundle_path: Path):
        self.bundle_path = Path(bundle_path)
        self.session = None
        self.preprocess_config = None
        self.policy_config = None
        
        self._load_configs()
        self._load_model()
    
    def _load_configs(self):
        """Load preprocessing and policy configs."""
        preprocess_path = self.bundle_path / "preprocess.json"
        policy_path = self.bundle_path / "policy.json"
        
        if preprocess_path.exists():
            with open(preprocess_path) as f:
                self.preprocess_config = json.load(f)
        else:
            self.preprocess_config = {}
        
        if policy_path.exists():
            with open(policy_path) as f:
                self.policy_config = json.load(f)
        else:
            self.policy_config = {
                "temperature": 1.0,
                "thresholds": {"SURVEILLANCE": 0.5, "TRUSTED": 0.6, "CONSERVATIVE": 0.7},
                "class_names": ["BPSK", "QPSK", "QAM16", "GFSK", "NOISE"],
            }
    
    def _load_model(self):
        """Load ONNX model."""
        model_path = self.bundle_path / "model.onnx"
        
        if not model_path.exists():
            print(f"WARNING: model.onnx not found at {model_path}")
            return
        
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(model_path))
            self.input_name = self.session.get_inputs()[0].name
        except ImportError:
            print("WARNING: onnxruntime not installed")
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess input data (STFT + normalization)."""
        # If already in feature format, just ensure correct shape
        if data.ndim == 4:
            return data.astype(np.float32)
        
        if data.ndim == 3:
            return data[np.newaxis, ...].astype(np.float32)
        
        # IQ data: compute STFT
        if np.iscomplexobj(data):
            from scipy.signal import stft
            cfg = self.preprocess_config.get("stft", {})
            n_fft = cfg.get("n_fft", 256)
            hop = cfg.get("hop_length", 64)
            
            _, _, Zxx = stft(data, nperseg=n_fft, noverlap=n_fft-hop)
            mag = np.abs(Zxx)
            mag_db = 20 * np.log10(mag + 1e-10)
            
            # Normalize
            mag_db = (mag_db - mag_db.mean()) / (mag_db.std() + 1e-10)
            
            return mag_db[np.newaxis, np.newaxis, ...].astype(np.float32)
        
        # Already preprocessed
        if data.ndim == 2:
            return data[np.newaxis, np.newaxis, ...].astype(np.float32)
        
        return data.astype(np.float32)
    
    def softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute calibrated softmax."""
        scaled = logits / temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)
    
    def predict(self, data: np.ndarray, mode: str = "TRUSTED") -> Dict[str, Any]:
        """Run full inference pipeline."""
        # Preprocess
        features = self.preprocess(data)
        
        # Inference
        if self.session is not None:
            logits = self.session.run(None, {self.input_name: features})[0]
        else:
            # Fallback: use pre-collected logits
            logits = self._fallback_inference(features)
        
        # Calibration
        T = self.policy_config.get("temperature", 1.0)
        probs = self.softmax(logits.astype(np.float32), T)
        
        # Prediction
        pred_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(np.max(probs, axis=1)[0])
        
        # Policy
        mode_upper = mode.upper()
        thresholds = self.policy_config.get("thresholds", {})
        tau = thresholds.get(mode_upper, 0.5)
        is_unknown = confidence < tau
        
        class_names = self.policy_config.get("class_names", [])
        pred_class = class_names[pred_idx] if pred_idx < len(class_names) else f"CLASS_{pred_idx}"
        
        return {
            "predicted_class": pred_class,
            "predicted_idx": pred_idx,
            "confidence": confidence,
            "mode": mode_upper,
            "threshold": tau,
            "is_unknown": is_unknown,
            "label": "UNKNOWN" if is_unknown else pred_class,
            "probs": probs[0].tolist(),
        }
    
    def _fallback_inference(self, features: np.ndarray) -> np.ndarray:
        """Fallback when ONNX model not available."""
        runs_v05 = project_root / "runs" / "v0_5"
        if runs_v05.exists():
            run_dirs = sorted([d for d in runs_v05.iterdir() if d.is_dir()])
            if run_dirs:
                logits_path = run_dirs[-1] / "logits_test_id.npz"
                if logits_path.exists():
                    data = np.load(logits_path)
                    return data["logits"][0:1].astype(np.float32)
        
        # Random fallback
        return np.random.randn(1, 5).astype(np.float32)


def benchmark_latency(bundle: EdgeBundle, n_runs: int = 200, 
                      warmup: int = 10) -> Dict[str, Any]:
    """Benchmark inference latency."""
    # Create synthetic input
    np.random.seed(42)
    test_input = np.random.randn(1, 1, 256, 15).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        bundle.predict(test_input)
    
    # Benchmark
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        bundle.predict(test_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    
    return {
        "n_runs": n_runs,
        "warmup": warmup,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "throughput_hz": float(1000 / np.mean(latencies)),
    }


def load_bundle(bundle_path: str = None) -> EdgeBundle:
    """Load edge bundle from path."""
    if bundle_path is None:
        bundle_path = project_root / "artifacts" / "bundle"
    return EdgeBundle(bundle_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run Edge Bundle — ONNX inference with policy"
    )
    parser.add_argument("--bundle", type=Path, default=None,
                        help="Bundle directory path")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input file (.npz, .iq, .npy)")
    parser.add_argument("--mode", type=str, default="TRUSTED",
                        choices=["SURVEILLANCE", "TRUSTED", "CONSERVATIVE",
                                 "surveillance", "trusted", "conservative"])
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark")
    parser.add_argument("--n_runs", type=int, default=200,
                        help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # cargamos el bundle
    bundle_path = args.bundle or (project_root / "artifacts" / "bundle")
    print(f"Bundle: {bundle_path}")
    
    bundle = EdgeBundle(bundle_path)
    
    if bundle.session is None:
        print("WARN: ONNX no disponible, usando fallback")  # TODO: mejorar fallback
    
    print(f"T={bundle.policy_config.get('temperature', 1.0):.2f}")
    
    # Load input
    if args.input:
        if args.input.suffix == ".npz":
            data = np.load(args.input)
            test_input = data.get("X", data.get("x", list(data.values())[0]))
        elif args.input.suffix == ".npy":
            test_input = np.load(args.input)
        else:
            test_input = np.fromfile(args.input, dtype=np.complex64)
    else:
        # Use demo sample
        demo_path = project_root / "demo_samples" / "id_sample.npz"
        if demo_path.exists():
            data = np.load(demo_path)
            test_input = data["X"]
        else:
            np.random.seed(42)
            test_input = np.random.randn(1, 1, 256, 15).astype(np.float32)
    
    print(f"Input shape: {test_input.shape}")
    print()
    
    # Run prediction
    result = bundle.predict(test_input, mode=args.mode)
    
    print("Prediction:")
    print(f"  Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Mode: {result['mode']}")
    print(f"  Threshold: {result['threshold']:.4f}")
    print(f"  Label: {result['label']}")
    print()
    
    # Benchmark if requested
    if args.benchmark:
        print(f"Running latency benchmark ({args.n_runs} runs)...")
        latency = benchmark_latency(bundle, n_runs=args.n_runs)
        
        print()
        print("Latency Results:")
        print(f"  p50: {latency['p50_ms']:.2f} ms")
        print(f"  p95: {latency['p95_ms']:.2f} ms")
        print(f"  p99: {latency['p99_ms']:.2f} ms")
        print(f"  mean: {latency['mean_ms']:.2f} ms (±{latency['std_ms']:.2f})")
        print(f"  throughput: {latency['throughput_hz']:.1f} Hz")
        
        # Save latency report
        reports_dir = project_root / "reports" / "metrics"
        reports_dir.mkdir(parents=True, exist_ok=True)
        latency_path = reports_dir / "latency.json"
        
        latency["timestamp"] = datetime.now().isoformat()
        latency["bundle_path"] = str(bundle_path)
        
        with open(latency_path, "w") as f:
            json.dump(latency, f, indent=2)
        
        print(f"\nSaved: {latency_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
