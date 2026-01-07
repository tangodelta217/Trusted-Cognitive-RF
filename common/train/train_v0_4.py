"""
Training script for V0.4 baseline model.

Trains a small CNN on cached features with early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
import csv
import random
import numpy as np
from datetime import datetime

from common.models.cnn_small import build_model_from_config, load_model_config
from common.train.dataset_cached import CachedFeaturesDataset


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_train_config(config_path: Path) -> Dict[str, Any]:
    """Load training configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
    
    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * X.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    
    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }


def train_model(
    train_config: Dict[str, Any],
    model_config: Dict[str, Any],
    run_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train the baseline model.
    
    Args:
        train_config: Training configuration.
        model_config: Model configuration.
        run_dir: Output directory for checkpoints and logs.
        verbose: Print progress.
        
    Returns:
        Training summary.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    seed = train_config["run"]["seed"]
    set_seed(seed)
    
    # Device
    device = torch.device("cpu")
    if train_config["device"].get("prefer_cuda", False) and torch.cuda.is_available():
        device = torch.device("cuda")
    
    if verbose:
        print(f"Device: {device}")
        print(f"Seed: {seed}")
    
    # Load datasets
    features_root = Path(train_config["data"]["features_cache_root"])
    train_ds = CachedFeaturesDataset(features_root / "train.npz")
    val_ds = CachedFeaturesDataset(features_root / "val.npz")
    
    if verbose:
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")
    
    # DataLoaders
    batch_size = train_config["train"]["batch_size"]
    num_workers = train_config["data"].get("num_workers", 0)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    
    # Build model
    model = build_model_from_config(model_config)
    model.to(device)
    
    if verbose:
        print(f"Model: {model_config['model_name']}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    lr = train_config["train"]["lr"]
    weight_decay = train_config["train"].get("weight_decay", 0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Early stopping
    early_stop_cfg = train_config["train"].get("early_stopping", {})
    early_stop_enabled = early_stop_cfg.get("enabled", False)
    patience = early_stop_cfg.get("patience", 10)
    
    # Training loop
    epochs = train_config["train"]["epochs"]
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    log_rows = []
    
    if verbose:
        print()
        print("Training...")
    
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        }
        log_rows.append(log_row)
        
        if verbose:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stop_enabled and patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    # Save last model
    torch.save(model.state_dict(), run_dir / "last_model.pt")
    
    # Save training log
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    
    # Save resolved config
    resolved_config = {
        "train_config": train_config,
        "model_config": model_config,
        "run_dir": str(run_dir),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.dump(resolved_config, f, default_flow_style=False)
    
    if verbose:
        print()
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Checkpoints saved to: {run_dir}")
    
    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_train_acc": log_rows[-1]["train_acc"],
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    # Quick test
    from pathlib import Path
    
    project_root = Path(__file__).parents[2]
    train_cfg = load_train_config(
        project_root / "common" / "train" / "configs" / "train_v0_4.yaml"
    )
    model_cfg = load_model_config(
        project_root / "common" / "models" / "configs" / "model_cnn_small_v0.yaml"
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / train_cfg["run"]["out_dir"] / f"run_{timestamp}"
    
    train_model(train_cfg, model_cfg, run_dir)
