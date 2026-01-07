"""
Evaluation script for V0.4 baseline model.

Evaluates on all splits and generates metrics report.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List
import yaml
import json
import csv
import numpy as np

from common.models.cnn_small import build_model_from_config
from common.train.dataset_cached import CachedFeaturesDataset
from common.eval.metrics import compute_metrics


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_predictions: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model on a single split.
    
    Returns:
        Dict with predictions, true labels, logits.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())
            if save_predictions:
                all_logits.append(logits.cpu().numpy())
    
    result = {
        "y_pred": np.concatenate(all_preds),
        "y_true": np.concatenate(all_labels),
    }
    
    if save_predictions:
        result["logits"] = np.concatenate(all_logits)
    
    return result


def evaluate_model(
    run_dir: Path,
    features_root: Path,
    splits: List[str] = None,
    save_predictions: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate trained model on multiple splits.
    
    Args:
        run_dir: Run directory containing best_model.pt and config.
        features_root: Path to cached features.
        splits: Splits to evaluate. Default: test_id, test_ood_mod, test_ood_chan.
        save_predictions: Save predictions NPZ files.
        verbose: Print progress.
        
    Returns:
        Evaluation report dict.
    """
    run_dir = Path(run_dir)
    features_root = Path(features_root)
    
    if splits is None:
        splits = ["test_id", "test_ood_mod", "test_ood_chan"]
    
    # Load config
    config_path = run_dir / "config_resolved.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model_config"]
    
    # Load model
    model = build_model_from_config(model_cfg)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    
    device = torch.device("cpu")
    model.to(device)
    
    # Class names
    class_names = ["BPSK", "QPSK", "QAM16", "GFSK", "NOISE", "PSK8", "QAM64", "CPFSK"]
    id_class_names = class_names[:5]
    
    report = {
        "run_dir": str(run_dir),
        "model_config": model_cfg,
        "splits": {}
    }
    
    for split in splits:
        features_path = features_root / f"{split}.npz"
        if not features_path.exists():
            if verbose:
                print(f"Skipping {split}: {features_path} not found")
            continue
        
        if verbose:
            print(f"Evaluating {split}...")
        
        dataset = CachedFeaturesDataset(features_path)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        result = evaluate_split(model, loader, device, save_predictions)
        
        # Compute metrics
        # For OOD splits, we use all classes in predictions but only ID classes exist in training
        metrics = compute_metrics(
            result["y_true"],
            result["y_pred"],
            num_classes=len(class_names),
            class_names=class_names
        )
        
        report["splits"][split] = {
            "n_examples": len(result["y_true"]),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "per_class": metrics["per_class"],
        }
        
        if verbose:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        
        # Save predictions
        if save_predictions:
            pred_path = run_dir / f"predictions_{split}.npz"
            np.savez_compressed(
                pred_path,
                y_true=result["y_true"],
                y_pred=result["y_pred"],
                logits=result.get("logits")
            )
    
    # Save report
    report_path = run_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    # Save confusion matrix for test_id as CSV
    if "test_id" in report["splits"]:
        cm = report["splits"]["test_id"]["confusion_matrix"]
        cm_path = run_dir / "confusion_matrix_test_id.csv"
        with open(cm_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([""] + id_class_names)
            # Rows
            for i, row in enumerate(cm[:5]):  # Only ID classes
                writer.writerow([id_class_names[i]] + row[:5])
        
        if verbose:
            print(f"\nConfusion matrix saved to: {cm_path}")
    
    if verbose:
        print(f"Report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    # Quick test
    import sys
    
    project_root = Path(__file__).parents[2]
    
    # Find most recent run
    runs_dir = project_root / "runs" / "v0_4"
    if runs_dir.exists():
        runs = sorted(runs_dir.iterdir())
        if runs:
            run_dir = runs[-1]
            features_root = project_root / "data" / "features" / "v0"
            
            evaluate_model(run_dir, features_root)
