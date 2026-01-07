"""
Plotting utilities for demo.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_spectrogram(
    features: NDArray[np.float32],
    title: str = "Spectrogram",
    save_path: Optional[Path] = None,
    show_colorbar: bool = True
) -> None:
    """
    Plot spectrogram from features.
    
    Args:
        features: Features array, shape (1, 256, 15) or (256, 15).
        title: Plot title.
        save_path: Path to save figure.
        show_colorbar: Whether to show colorbar.
    """
    # Handle channel dimension
    if features.ndim == 3:
        features = features.squeeze(0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(
        features,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Frequency Bin")
    ax.set_title(title)
    
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Log Power (normalized)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_demo_result(
    features: NDArray[np.float32],
    top_k: List[Dict[str, float]],
    predicted_class: str,
    confidence: float,
    is_unknown: bool,
    mode: str,
    tau: float,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot comprehensive demo result.
    
    Args:
        features: Spectrogram features.
        top_k: Top-k predictions list.
        predicted_class: Predicted class name.
        confidence: Confidence score.
        is_unknown: Whether marked as UNKNOWN.
        mode: Operating mode.
        tau: Threshold.
        save_path: Path to save.
    """
    if features.ndim == 3:
        features = features.squeeze(0)
    
    fig = plt.figure(figsize=(12, 5))
    
    # Spectrogram
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(features, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_xlabel("Time Frame")
    ax1.set_ylabel("Frequency Bin")
    ax1.set_title("Input Spectrogram")
    plt.colorbar(im, ax=ax1)
    
    # Prediction bar chart
    ax2 = fig.add_subplot(1, 2, 2)
    classes = [p["class"] for p in top_k]
    probs = [p["prob"] for p in top_k]
    
    colors = ['coral' if is_unknown else 'steelblue'] * len(classes)
    bars = ax2.barh(classes, probs, color=colors)
    
    # Mark threshold
    ax2.axvline(tau, color='red', linestyle='--', linewidth=2, label=f'τ={tau:.3f}')
    
    ax2.set_xlabel("Probability")
    ax2.set_xlim(0, 1)
    ax2.set_title(f"Prediction ({mode})")
    
    # Add confidence annotation
    status = "UNKNOWN" if is_unknown else predicted_class
    ax2.annotate(
        f"conf={confidence:.3f}\n→ {status}",
        xy=(0.95, 0.05), xycoords='axes fraction',
        ha='right', va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
