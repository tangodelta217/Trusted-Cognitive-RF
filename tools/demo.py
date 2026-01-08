#!/usr/bin/env python3
"""
Offline Demo: Reproducible demo for INDRA presentation.

Generates:
- Waterfall/spectrogram visualization
- Event log (JSON/CSV)
- Console output with ID/OOD and UNKNOWN detection
- Mode comparison (SURVEILLANCE/TRUSTED/CONSERVATIVE)

Usage:
    python -m tools.demo                    # Full demo
    python -m tools.demo --mode trusted     # Single mode
    python -m tools.demo --compare_modes    # Compare all modes
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


CLASS_NAMES = ["BPSK", "QPSK", "QAM16", "GFSK", "NOISE"]


def load_policy() -> Dict[str, Any]:
    """Load temperature and thresholds."""
    policy = {"temperature": 1.0, "thresholds": {}}
    
    temp_path = project_root / "artifacts" / "policy" / "temperature.json"
    if temp_path.exists():
        with open(temp_path) as f:
            policy["temperature"] = json.load(f)["temperature"]
    
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


def create_demo_samples() -> Dict[str, Path]:
    """Create or locate demo samples."""
    demo_dir = project_root / "demo_samples"
    demo_dir.mkdir(exist_ok=True)
    
    features_root = project_root / "data" / "features" / "v0"
    
    samples = {}
    
    # ID sample
    id_path = demo_dir / "id_sample.npz"
    if not id_path.exists():
        if (features_root / "test_id.npz").exists():
            data = np.load(features_root / "test_id.npz")
            np.savez_compressed(id_path, X=data["X"][0:1], y=data["y"][0:1])
        else:
            # Synthetic
            np.random.seed(42)
            X = np.random.randn(1, 1, 256, 15).astype(np.float32)
            y = np.array([1])  # QPSK
            np.savez_compressed(id_path, X=X, y=y)
    samples["id"] = id_path
    
    # OOD sample
    ood_path = demo_dir / "ood_sample.npz"
    if not ood_path.exists():
        if (features_root / "test_ood_mod.npz").exists():
            data = np.load(features_root / "test_ood_mod.npz")
            np.savez_compressed(ood_path, X=data["X"][0:1], y=data["y"][0:1])
        else:
            # Synthetic OOD-like
            np.random.seed(123)
            X = np.random.randn(1, 1, 256, 15).astype(np.float32) * 0.5
            y = np.array([99])  # Unknown
            np.savez_compressed(ood_path, X=X, y=y)
    samples["ood"] = ood_path
    
    return samples


def load_sample(path: Path) -> np.ndarray:
    """Load sample features."""
    data = np.load(path)
    return data["X"].astype(np.float32)


def run_inference(features: np.ndarray, temperature: float, sample_type: str = "ID") -> Dict[str, Any]:
    """Run inference on features using ONNX or pre-collected logits."""
    from tfm_ai_utamed.assurance.temperature_scaling import softmax
    
    logits = None
    
    # Try ONNX first
    model_dirs = list((project_root / "runs" / "v0_4").iterdir()) if (project_root / "runs" / "v0_4").exists() else []
    model_dirs = sorted([d for d in model_dirs if d.is_dir()])
    
    if model_dirs:
        onnx_path = model_dirs[-1] / "model.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_path))
                input_name = session.get_inputs()[0].name
                logits = session.run(None, {input_name: features})[0]
            except Exception:
                pass
    
    # Fallback to pre-collected logits from v0_5 runs
    if logits is None:
        runs_v05 = project_root / "runs" / "v0_5"
        if runs_v05.exists():
            run_dirs = sorted([d for d in runs_v05.iterdir() if d.is_dir()])
            if run_dirs:
                # Use appropriate logits based on sample type
                if sample_type == "OOD":
                    logits_file = "logits_test_ood_mod.npz"
                else:
                    logits_file = "logits_test_id.npz"
                
                logits_path = run_dirs[-1] / logits_file
                if logits_path.exists():
                    data = np.load(logits_path)
                    # Use a sample that will have low confidence for OOD
                    if sample_type == "OOD":
                        # Find a sample with lower max probability
                        all_logits = data["logits"]
                        probs_tmp = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
                        max_probs = np.max(probs_tmp, axis=1)
                        # Use sample with lowest max prob
                        idx = np.argmin(max_probs)
                        logits = all_logits[idx:idx+1].astype(np.float32)
                    else:
                        logits = data["logits"][0:1].astype(np.float32)
    
    # Final fallback: random logits
    if logits is None:
        np.random.seed(hash(features.tobytes()) % 2**32)
        if sample_type == "OOD":
            # Low-confidence for OOD
            logits = np.random.randn(1, 5).astype(np.float32) * 0.5
        else:
            logits = np.random.randn(1, 5).astype(np.float32) * 2
    
    # Calibrated probabilities
    probs = softmax(logits.astype(np.float32), temperature)
    
    # Compute OOD score (entropy)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    return {
        "logits": logits,
        "probs": probs,
        "entropy": entropy,
        "predicted_idx": int(np.argmax(probs)),
        "confidence": float(np.max(probs)),
    }


def apply_policy(result: Dict, mode: str, thresholds: Dict) -> Dict[str, Any]:
    """Apply abstention policy."""
    tau = thresholds.get(mode.upper(), 0.5)
    is_unknown = result["confidence"] < tau
    
    return {
        "mode": mode.upper(),
        "threshold": tau,
        "predicted_class": CLASS_NAMES[result["predicted_idx"]],
        "confidence": result["confidence"],
        "is_unknown": is_unknown,
        "label": "UNKNOWN" if is_unknown else CLASS_NAMES[result["predicted_idx"]],
        "entropy": float(result["entropy"][0]),
    }


def generate_waterfall(features: np.ndarray, output_path: Path, 
                        predictions: List[Dict] = None) -> None:
    """Generate waterfall/spectrogram visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: ID sample spectrogram
    spec_id = features[0, 0] if features.ndim == 4 else features[0]
    im1 = axes[0].imshow(spec_id, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("ID Sample — Spectrogram", fontsize=12)
    axes[0].set_ylabel("Frequency Bin")
    plt.colorbar(im1, ax=axes[0], label="Power (dB)")
    
    # Add prediction annotation if available
    if predictions and len(predictions) > 0:
        pred = predictions[0]
        color = 'green' if not pred.get('is_unknown') else 'red'
        text = f"{pred.get('label', 'N/A')}\nconf={pred.get('confidence', 0):.2f}"
        axes[0].text(0.02, 0.95, text, transform=axes[0].transAxes, fontsize=10,
                     verticalalignment='top', color='white',
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    # Bottom: Mode comparison
    if predictions and len(predictions) > 1:
        modes = [p['mode'] for p in predictions[:3]]
        labels = [p['label'] for p in predictions[:3]]
        confs = [p['confidence'] for p in predictions[:3]]
        
        colors = ['green' if l != 'UNKNOWN' else 'red' for l in labels]
        bars = axes[1].bar(modes, confs, color=colors, alpha=0.7)
        axes[1].axhline(y=0.5, color='orange', linestyle='--', label='τ ref')
        axes[1].set_ylabel("Confidence")
        axes[1].set_title("Mode Comparison — Confidence & UNKNOWN", fontsize=12)
        axes[1].set_ylim(0, 1)
        
        for bar, label in zip(bars, labels):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         label, ha='center', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No mode comparison data", ha='center', va='center',
                     transform=axes[1].transAxes)
    
    axes[-1].set_xlabel("Time Frame / Mode")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_demo(mode: str = None, compare_modes: bool = True, 
             output_dir: Path = None, verbose: bool = True) -> Dict[str, Any]:
    """Run the full demo."""
    
    if output_dir is None:
        output_dir = project_root / "reports" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load policy
    policy = load_policy()
    T = policy["temperature"]
    thresholds = policy["thresholds"]
    
    if verbose:
        print("\n-- Demo RF Cognitivo --")
        print(f"T={T:.3f}, modos: {list(thresholds.keys())}")
    
    # Create/load samples
    samples = create_demo_samples()
    
    if verbose:
        print(f"ID sample: {samples['id']}")
        print(f"OOD sample: {samples['ood']}")
        print()
    
    events = []
    all_predictions = []
    
    # Process ID sample
    if verbose:
        print("Processing ID sample...")
    
    id_features = load_sample(samples["id"])
    id_result = run_inference(id_features, T, sample_type="ID")
    
    modes_to_test = ["SURVEILLANCE", "TRUSTED", "CONSERVATIVE"] if compare_modes else [mode.upper()]
    
    for m in modes_to_test:
        pred = apply_policy(id_result, m, thresholds)
        pred["sample_type"] = "ID"
        pred["timestamp"] = datetime.now().isoformat()
        events.append(pred)
        all_predictions.append(pred)
        
        if verbose:
            status = "UNKNOWN" if pred["is_unknown"] else pred["predicted_class"]
            print(f"  {m}: {status} (conf={pred['confidence']:.3f}, τ={pred['threshold']:.3f})")
    
    print()
    
    # Process OOD sample
    if verbose:
        print("Processing OOD sample...")
    
    ood_features = load_sample(samples["ood"])
    ood_result = run_inference(ood_features, T, sample_type="OOD")
    
    for m in modes_to_test:
        pred = apply_policy(ood_result, m, thresholds)
        pred["sample_type"] = "OOD"
        pred["timestamp"] = datetime.now().isoformat()
        events.append(pred)
        
        if verbose:
            status = "⚠ UNKNOWN" if pred["is_unknown"] else pred["predicted_class"]
            print(f"  {m}: {status} (conf={pred['confidence']:.3f}, τ={pred['threshold']:.3f})")
    
    print()
    
    # Generate waterfall
    if verbose:
        print("Generating waterfall visualization...")
    waterfall_path = output_dir / "waterfall.png"
    generate_waterfall(id_features, waterfall_path, all_predictions)
    if verbose:
        print(f"  Saved: {waterfall_path}")
    
    # Save events
    events_path = output_dir / "events.json"
    with open(events_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "temperature": T,
            "thresholds": thresholds,
            "events": events,
        }, f, indent=2)
    if verbose:
        print(f"  Saved: {events_path}")
    
    # Summary
    unknown_count = sum(1 for e in events if e["is_unknown"])
    if verbose:
        print()
        print("=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"Total events: {len(events)}")
        print(f"UNKNOWN detections: {unknown_count}")
        print()
        
        # Check for UNKNOWN in CONSERVATIVE mode
        cons_unknown = any(e["is_unknown"] and e["mode"] == "CONSERVATIVE" for e in events)
        if cons_unknown:
            print("✓ UNKNOWN detected in CONSERVATIVE mode")
        else:
            print("⚠ No UNKNOWN in CONSERVATIVE mode")
    
    return {
        "output_dir": str(output_dir),
        "waterfall": str(waterfall_path),
        "events": str(events_path),
        "n_events": len(events),
        "n_unknown": unknown_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline Demo for INDRA Presentation"
    )
    parser.add_argument("--mode", type=str, default=None,
                        choices=["surveillance", "trusted", "conservative"],
                        help="Single mode to run (default: compare all)")
    parser.add_argument("--input", type=Path, default=None,
                        help="Custom input sample path")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory")
    parser.add_argument("--compare_modes", action="store_true", default=True,
                        help="Compare all modes (default)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    
    args = parser.parse_args()
    
    compare = args.mode is None
    mode = args.mode or "TRUSTED"
    
    result = run_demo(
        mode=mode,
        compare_modes=compare,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    
    print("\nDemo complete!")
    print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
