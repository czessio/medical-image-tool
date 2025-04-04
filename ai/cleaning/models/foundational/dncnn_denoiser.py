"""
DnCNN denoiser model implementation.
Implements the classic DnCNN architecture for medical image denoising.
"""
import os
import logging
import numpy as np
from typing import Dict, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.torch_model import TorchModel
from ai.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class DnCNN(nn.Module):
    """
    DnCNN model architecture for image denoising.
    
    Based on the paper "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
    by Zhang et al. (2017).
    """
    
    def __init__(self, in_channels=1, out_channels=1, num_layers=17, features=64):
        """
        Initialize the DnCNN model.
        
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            out_channels: Number of output channels (usually the same as in_channels)
            num_layers: Number of convolutional layers
            features: Number of feature maps in each layer
        """
        super(DnCNN, self).__init__()
        
        # First layer: Conv+ReLU
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Middle layers: Conv+BN+ReLU
        self.middle_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.middle_layers.append(
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True)
                )
            )
            
        # Last layer: Conv
        self.last_layer = nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=False)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Forward pass through the network.
        DnCNN learns the noise component, so we subtract it from the input.
        
        Args:
            x: Input noisy image tensor
            
        Returns:
            Denoised image tensor
        """
        # Extract noise component
        residual = self.first_layer(x)
        for layer in self.middle_layers:
            residual = layer(residual)
        residual = self.last_layer(residual)
        
        # Subtract the noise from the input (residual learning)
        return x - residual


class DnCNNDenoiser(TorchModel):
    """
    DnCNN-based denoiser model.
    
    Uses the DnCNN architecture which focuses on residual learning to
    estimate the noise component and then subtract it from the noisy image.
    """
    
    def __init__(self, model_path=None, device=None, num_layers=17, in_channels=1, out_channels=1):
        """
        Initialize the DnCNN denoiser model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            num_layers: Number of layers in the DnCNN model
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            out_channels: Number of output channels (usually same as in_channels)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        super().__init__(model_path, device)
    
    def _create_model_architecture(self):
        """Create the DnCNN model architecture."""
        model = DnCNN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            features=64
        )
        
        return model
    
    def preprocess(self, image, noise_level=None):
        """
        Preprocess the input image for the DnCNN model.
        
        Args:
            image: Input image as numpy array
            noise_level: Optional noise level to add for testing
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Use parent preprocessing but ensure channels are correct
        tensor = super().preprocess(image)
        
        # If input is RGB but model expects grayscale, convert to grayscale
        if tensor.shape[1] == 3 and self.in_channels == 1:
            tensor = tensor.mean(dim=1, keepdim=True)
        
        # If input is grayscale but model expects RGB, repeat channels
        if tensor.shape[1] == 1 and self.in_channels == 3:
            tensor = tensor.repeat(1, 3, 1, 1)
        
        # Add synthetic noise for testing if specified
        if noise_level is not None:
            noise = torch.randn_like(tensor) * noise_level
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0, 1)
        
        return tensor
    
    def inference(self, preprocessed_tensor):
        """
        Run inference with the DnCNN model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Denoised output tensor
        """
        with torch.no_grad():
            return self.model(preprocessed_tensor)
    
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output.
        
        Args:
            model_output: Raw output from the model
            original_image: Original input image (if needed for reference)
            
        Returns:
            numpy.ndarray: Denoised image as numpy array
        """
        return super().postprocess(model_output, original_image)
    
    def process(self, image, noise_level=None):
        """
        Process an image with the DnCNN model.
        
        Args:
            image: Input image as numpy array
            noise_level: Optional noise level to add for testing
            
        Returns:
            numpy.ndarray: Denoised output image
        """
        if not self.initialized:
            self.initialize()
        
        preprocessed = self.preprocess(image, noise_level)
        output = self.inference(preprocessed)
        result = self.postprocess(output, image)
        
        return result

# Register the model
ModelRegistry.register("dncnn_denoiser", DnCNNDenoiser)