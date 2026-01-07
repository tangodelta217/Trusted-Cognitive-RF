"""
Small CNN model for RF signal classification.

Designed to be quantization-friendly with simple ops:
Conv2d + BatchNorm2d + ReLU + MaxPool2d + Linear
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import yaml


def load_model_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load model configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "model_cnn_small_v0.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BatchNorm (optional) -> ReLU -> MaxPool."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        pool_size: tuple = (2, 1)
    ):
        super().__init__()
        
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if pool_size[0] > 1 or pool_size[1] > 1:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNSmall(nn.Module):
    """
    Small CNN for RF signal classification.
    
    Architecture:
    - 3 ConvBlocks with increasing channels
    - Global average pooling
    - FC layers with dropout
    
    Designed to be quantization-friendly.
    """
    
    def __init__(
        self,
        input_shape: tuple = (1, 256, 15),
        num_classes: int = 5,
        conv_channels: List[int] = None,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        pool_size: tuple = (2, 1),
        fc_hidden: int = 128,
        dropout_p: float = 0.2
    ):
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [8, 16, 32]
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Build convolutional layers
        in_channels = input_shape[0]
        conv_layers = []
        
        for out_channels in conv_channels:
            conv_layers.append(
                ConvBlock(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    use_batchnorm=use_batchnorm,
                    pool_size=pool_size
                )
            )
            in_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate feature size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy)
            self.flat_features = conv_out.view(1, -1).shape[1]
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_features, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_hidden, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (B, C, F, T) = (B, 1, 256, 15)
            
        Returns:
            Logits, shape (B, num_classes)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities via softmax."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def build_model_from_config(config: Dict[str, Any]) -> CNNSmall:
    """
    Build CNNSmall model from configuration dict.
    
    Args:
        config: Model configuration.
        
    Returns:
        CNNSmall model instance.
    """
    arch = config.get("architecture", {})
    
    return CNNSmall(
        input_shape=tuple(config.get("input_shape", [1, 256, 15])),
        num_classes=config.get("num_classes", 5),
        conv_channels=arch.get("conv_channels", [8, 16, 32]),
        kernel_size=arch.get("kernel_size", [3, 3])[0] if isinstance(arch.get("kernel_size"), list) else arch.get("kernel_size", 3),
        use_batchnorm=arch.get("use_batchnorm", True),
        pool_size=tuple(arch.get("pool", [2, 1])),
        fc_hidden=arch.get("fc_hidden", 128),
        dropout_p=arch.get("dropout_p", 0.2)
    )


if __name__ == "__main__":
    # Quick test
    config = load_model_config()
    model = build_model_from_config(config)
    
    print(f"Model: {config['model_name']}")
    print(f"Input shape: {config['input_shape']}")
    print(f"Num classes: {config['num_classes']}")
    print()
    print(model)
    print()
    
    # Test forward pass
    x = torch.randn(4, 1, 256, 15)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Flat features: {model.flat_features}")
