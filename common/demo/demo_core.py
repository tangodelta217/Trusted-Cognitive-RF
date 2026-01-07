"""
Demo core pipeline: load → features → infer → policy → result.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time
import json


@dataclass
class DemoResult:
    """Result of demo pipeline."""
    
    # Input info
    split: str
    index: int
    mode: str
    
    # Calibration params
    temperature: float
    tau: float
    
    # Prediction
    predicted_class: str
    confidence: float
    is_unknown: bool
    top_k: List[Dict[str, float]]
    
    # Features
    features: NDArray[np.float32] = field(repr=False)
    
    # Latencies (ms)
    latency_load_ms: float = 0.0
    latency_features_ms: float = 0.0
    latency_infer_ms: float = 0.0
    latency_policy_ms: float = 0.0
    latency_total_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "split": self.split,
            "index": self.index,
            "mode": self.mode,
            "temperature": self.temperature,
            "tau": self.tau,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "is_unknown": self.is_unknown,
            "top_k": self.top_k,
            "latency_ms": {
                "load": round(self.latency_load_ms, 2),
                "features": round(self.latency_features_ms, 2),
                "infer": round(self.latency_infer_ms, 2),
                "policy": round(self.latency_policy_ms, 2),
                "total": round(self.latency_total_ms, 2),
            }
        }


def softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    """Numerically stable softmax."""
    x = logits - np.max(logits)
    exp_x = np.exp(x)
    return (exp_x / exp_x.sum()).astype(np.float32)


def run_demo_pipeline(
    split: str,
    index: int,
    mode: str,
    dataset_root: Path,
    features_root: Optional[Path],
    model_path: Path,
    operating_points: Dict[str, Any],
    use_features_cache: bool = True,
    class_names: List[str] = None,
    top_k: int = 3
) -> DemoResult:
    """
    Run the complete demo pipeline.
    
    Args:
        split: Dataset split name.
        index: Example index.
        mode: Operating mode (SURVEILLANCE, TRUSTED, CONSERVATIVE).
        dataset_root: Path to IQ dataset.
        features_root: Path to cached features (optional).
        model_path: Path to ONNX model.
        operating_points: Operating points dict.
        use_features_cache: Whether to use cached features.
        class_names: List of class names.
        top_k: Number of top predictions to return.
        
    Returns:
        DemoResult with all info.
    """
    from common.infer.onnx_infer import ONNXInferenceSession
    from common.policy.operating_points import get_preset, get_temperature, CLASS_NAMES
    
    if class_names is None:
        class_names = CLASS_NAMES
    
    total_start = time.perf_counter()
    
    # === 1. Load data ===
    load_start = time.perf_counter()
    
    if use_features_cache and features_root is not None:
        features_path = features_root / f"{split}.npz"
        if features_path.exists():
            data = np.load(features_path)
            features = data["X"][index]
            y_true = data["y"][index]
        else:
            use_features_cache = False
    
    if not use_features_cache:
        # Load IQ and extract features
        iq_path = dataset_root / f"{split}.npz"
        data = np.load(iq_path, allow_pickle=True)
        iq = data["iq"][index]
        y_true = data["y"][index]
        
        # Extract features
        from common.features import extract_features, load_config
        feat_start = time.perf_counter()
        config = load_config()
        features = extract_features(iq, config)
        latency_features_ms = (time.perf_counter() - feat_start) * 1000
    else:
        latency_features_ms = 0.0
    
    latency_load_ms = (time.perf_counter() - load_start) * 1000 - latency_features_ms
    
    # === 2. Inference ===
    infer_start = time.perf_counter()
    session = ONNXInferenceSession(model_path)
    logits, _ = session.predict_with_latency(features)
    logits = logits.squeeze()  # (5,)
    latency_infer_ms = (time.perf_counter() - infer_start) * 1000
    
    # === 3. Calibration + Policy ===
    policy_start = time.perf_counter()
    
    T = get_temperature(operating_points)
    preset = get_preset(operating_points, mode)
    tau = preset["tau"]
    
    # Temperature scaling
    logits_cal = logits / T
    probs = softmax(logits_cal)
    
    # Confidence and prediction
    confidence = float(np.max(probs))
    pred_idx = int(np.argmax(probs))
    predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else f"CLASS_{pred_idx}"
    
    # UNKNOWN detection
    is_unknown = confidence < tau
    
    # Top-k predictions
    sorted_indices = np.argsort(-probs)[:top_k]
    top_k_list = [
        {"class": class_names[i] if i < len(class_names) else f"CLASS_{i}", 
         "prob": float(probs[i])}
        for i in sorted_indices
    ]
    
    latency_policy_ms = (time.perf_counter() - policy_start) * 1000
    
    latency_total_ms = (time.perf_counter() - total_start) * 1000
    
    return DemoResult(
        split=split,
        index=index,
        mode=mode,
        temperature=T,
        tau=tau,
        predicted_class=predicted_class,
        confidence=confidence,
        is_unknown=is_unknown,
        top_k=top_k_list,
        features=features,
        latency_load_ms=latency_load_ms,
        latency_features_ms=latency_features_ms,
        latency_infer_ms=latency_infer_ms,
        latency_policy_ms=latency_policy_ms,
        latency_total_ms=latency_total_ms,
    )
