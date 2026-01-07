"""
Export PyTorch model to ONNX format.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import yaml


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 1, 256, 15),
    opset_version: int = 11,
    dynamic_batch: bool = True,
    verbose: bool = True
) -> Path:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model.
        output_path: Output ONNX file path.
        input_shape: Input tensor shape (B, C, F, T).
        opset_version: ONNX opset version.
        dynamic_batch: Allow dynamic batch size.
        verbose: Print progress.
        
    Returns:
        Path to exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Dynamic axes for batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    
    # Export using legacy exporter for compatibility
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False  # Use legacy exporter
    )
    
    if verbose:
        print(f"Exported ONNX model to: {output_path}")
        print(f"  Input shape: {input_shape}")
        print(f"  Opset version: {opset_version}")
        print(f"  Dynamic batch: {dynamic_batch}")
    
    return output_path


def export_from_checkpoint(
    checkpoint_path: Path,
    model_config_path: Path,
    output_path: Path,
    verbose: bool = True
) -> Path:
    """
    Export model from checkpoint to ONNX.
    
    Args:
        checkpoint_path: Path to .pt checkpoint.
        model_config_path: Path to model config YAML.
        output_path: Output ONNX path.
        verbose: Print progress.
        
    Returns:
        Path to exported ONNX file.
    """
    from common.models.cnn_small import build_model_from_config, load_model_config
    
    # Load config
    model_cfg = load_model_config(model_config_path)
    
    # Build model
    model = build_model_from_config(model_cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    
    # Get input shape from config
    input_shape = tuple(model_cfg.get("input_shape", [1, 256, 15]))
    batch_input_shape = (1,) + input_shape
    
    return export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=batch_input_shape,
        verbose=verbose
    )


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    project_root = Path(__file__).parents[2]
    
    # Find latest run
    runs_dir = project_root / "runs" / "v0_4"
    if runs_dir.exists():
        runs = sorted(runs_dir.iterdir())
        if runs:
            run_dir = runs[-1]
            
            export_from_checkpoint(
                checkpoint_path=run_dir / "best_model.pt",
                model_config_path=project_root / "common" / "models" / "configs" / "model_cnn_small_v0.yaml",
                output_path=run_dir / "model.onnx"
            )
