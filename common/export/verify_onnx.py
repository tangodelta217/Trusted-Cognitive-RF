"""
Verify ONNX model parity with PyTorch model.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml


def verify_onnx_parity(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    test_inputs: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify ONNX model produces same outputs as PyTorch model.
    
    Args:
        pytorch_model: PyTorch model.
        onnx_path: Path to ONNX model.
        test_inputs: Test input array, shape (N, C, F, T).
        rtol: Relative tolerance for allclose.
        atol: Absolute tolerance for allclose.
        verbose: Print progress.
        
    Returns:
        Verification result dict.
    """
    import onnxruntime as ort
    
    onnx_path = Path(onnx_path)
    
    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Get input/output names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pt_input = torch.from_numpy(test_inputs).float()
        pt_output = pytorch_model(pt_input).numpy()
    
    # ONNX inference
    onnx_output = ort_session.run(
        [output_name],
        {input_name: test_inputs.astype(np.float32)}
    )[0]
    
    # Compare outputs
    max_diff = np.max(np.abs(pt_output - onnx_output))
    mean_diff = np.mean(np.abs(pt_output - onnx_output))
    
    # Check class predictions match
    pt_preds = np.argmax(pt_output, axis=1)
    onnx_preds = np.argmax(onnx_output, axis=1)
    pred_match_rate = np.mean(pt_preds == onnx_preds)
    
    # Check allclose
    values_close = np.allclose(pt_output, onnx_output, rtol=rtol, atol=atol)
    
    result = {
        "passed": values_close and pred_match_rate >= 0.99,
        "n_examples": len(test_inputs),
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "pred_match_rate": float(pred_match_rate),
        "values_close": values_close,
        "rtol": rtol,
        "atol": atol,
    }
    
    if verbose:
        status = "✓ PASSED" if result["passed"] else "✗ FAILED"
        print(f"ONNX Parity Check: {status}")
        print(f"  Examples tested: {result['n_examples']}")
        print(f"  Max logit diff: {result['max_diff']:.2e}")
        print(f"  Mean logit diff: {result['mean_diff']:.2e}")
        print(f"  Prediction match rate: {result['pred_match_rate']:.4f}")
        print(f"  Values allclose: {result['values_close']}")
    
    return result


def verify_from_run(
    run_dir: Path,
    features_root: Path,
    split: str = "test_id",
    n_examples: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify ONNX parity using a run directory and cached features.
    
    Args:
        run_dir: Run directory with model.onnx and config.
        features_root: Path to cached features.
        split: Split to use for verification.
        n_examples: Number of examples to test.
        verbose: Print progress.
        
    Returns:
        Verification result.
    """
    from common.models.cnn_small import build_model_from_config
    from common.train.dataset_cached import CachedFeaturesDataset
    
    run_dir = Path(run_dir)
    features_root = Path(features_root)
    
    # Load config
    with open(run_dir / "config_resolved.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model_config"]
    
    # Load PyTorch model
    model = build_model_from_config(model_cfg)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    
    # Load test data
    dataset = CachedFeaturesDataset(features_root / f"{split}.npz")
    
    # Get subset of examples
    n_examples = min(n_examples, len(dataset))
    test_inputs = dataset.X[:n_examples]
    
    # Verify
    onnx_path = run_dir / "model.onnx"
    
    return verify_onnx_parity(
        pytorch_model=model,
        onnx_path=onnx_path,
        test_inputs=test_inputs,
        verbose=verbose
    )


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parents[2]
    
    runs_dir = project_root / "runs" / "v0_4"
    if runs_dir.exists():
        runs = sorted(runs_dir.iterdir())
        if runs:
            run_dir = runs[-1]
            features_root = project_root / "data" / "features" / "v0"
            
            verify_from_run(run_dir, features_root)
