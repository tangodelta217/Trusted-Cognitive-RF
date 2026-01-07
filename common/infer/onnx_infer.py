"""
ONNX Runtime inference for RF classification.

Falls back to PyTorch if ONNXRuntime not available.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional
import time


# Try to import onnxruntime, fall back to PyTorch
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXInferenceSession:
    """
    ONNX Runtime inference session for RF model.
    Falls back to PyTorch if ONNX Runtime not available.
    """
    
    def __init__(self, model_path: Path):
        """
        Initialize inference session.
        
        Args:
            model_path: Path to .onnx or .pt model file.
        """
        self.model_path = Path(model_path)
        self.use_onnx = ONNX_AVAILABLE and self.model_path.suffix == ".onnx"
        
        if self.use_onnx:
            self._init_onnx()
        else:
            self._init_pytorch()
    
    def _init_onnx(self):
        """Initialize ONNX session."""
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def _init_pytorch(self):
        """Initialize PyTorch model."""
        import torch
        from common.models.cnn_small import build_model_from_config, load_model_config
        
        # Find matching .pt file
        pt_path = self.model_path.with_suffix(".pt") if self.model_path.suffix == ".onnx" else self.model_path
        if not pt_path.exists():
            pt_path = self.model_path.parent / "best_model.pt"
        
        if not pt_path.exists():
            raise FileNotFoundError(f"No model found: {self.model_path}")
        
        # Load model
        project_root = Path(__file__).parents[2]
        config_path = project_root / "common" / "models" / "configs" / "model_cnn_small_v0.yaml"
        model_cfg = load_model_config(config_path)
        
        self.model = build_model_from_config(model_cfg)
        self.model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        self.model.eval()
    
    def predict(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Run inference.
        
        Args:
            x: Input features, shape (1, 1, 256, 15) or (1, 256, 15).
            
        Returns:
            Logits, shape (1, 5).
        """
        if x.ndim == 3:
            x = x[np.newaxis, ...]
        x = x.astype(np.float32)
        
        if self.use_onnx:
            outputs = self.session.run([self.output_name], {self.input_name: x})
            return outputs[0]
        else:
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(x)
                logits = self.model(tensor)
                return logits.numpy()
    
    def predict_with_latency(self, x: NDArray[np.float32]) -> tuple:
        """Run inference and measure latency."""
        start = time.perf_counter()
        logits = self.predict(x)
        latency_ms = (time.perf_counter() - start) * 1000
        return logits, latency_ms


def find_model_onnx(run_dir: Optional[Path] = None) -> Path:
    """Find model.onnx (or best_model.pt) in a run directory."""
    if run_dir is None:
        project_root = Path(__file__).parents[2]
        runs_v0_4 = project_root / "runs" / "v0_4"
        if runs_v0_4.exists():
            runs = sorted([r for r in runs_v0_4.iterdir() if r.is_dir()])
            if runs:
                run_dir = runs[-1]
    
    if run_dir is None:
        raise FileNotFoundError("No V0.4 run found")
    
    model_path = run_dir / "model.onnx"
    if not model_path.exists():
        model_path = run_dir / "best_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {run_dir}")
    
    return model_path
