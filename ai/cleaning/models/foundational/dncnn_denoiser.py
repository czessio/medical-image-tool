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
    
    This implementation exactly matches the structure in dncnn_25.pth which has numbered
    layers in the format "model.0.weight", "model.0.bias", etc.
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
        
        # Sequential model that matches the dncnn_25.pth structure
        self.model = nn.Sequential()
        
        # First layer: Conv + ReLU
        self.model.add_module('0', nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True))
        self.model.add_module('1', nn.ReLU(inplace=True))
        
        # Middle layers: Conv+BN+ReLU
        for i in range(1, num_layers-1):
            self.model.add_module(f'{i*2}', nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True))
            self.model.add_module(f'{i*2+1}', nn.ReLU(inplace=True))
            
        # Last layer: Conv (final, no activation)
        self.model.add_module(f'{(num_layers-1)*2}', nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=True))
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
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
        # Estimate the noise component
        noise = self.model(x)
        
        # Subtract the noise from the input (residual learning)
        return x - noise


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
        
        # Analyze the model path to determine parameters
        if model_path and os.path.basename(model_path) == "dncnn_25.pth":
            # This specific model has 17 layers and operates on grayscale
            self.num_layers = 17
            self.in_channels = 1
            self.out_channels = 1
            logger.info("Loading dncnn_25.pth - using 17 layers and grayscale format")
        else:
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
    
    def _custom_load_state_dict(self, state_dict):
        """
        Custom loading function for matching dncnn_25.pth weight file structure.
        
        Args:
            state_dict: State dictionary from the weight file
            
        Returns:
            bool: True if loading succeeded, False otherwise
        """
        logger.info("Using custom weight loading for DnCNN model")
        
        try:
            # The dncnn_25.pth file has keys that match our model exactly,
            # so we can load the state dict directly
            self.model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error in weight loading: {e}")
            
            # Try to show what keys are available
            if state_dict:
                logger.info(f"Keys in state_dict: {list(state_dict.keys())[:5]}...")
                
            # Try non-strict loading as a fallback
            try:
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Model weights loaded with non-strict matching")
                return True
            except Exception as e2:
                logger.error(f"Error in fallback non-strict loading: {e2}")
                return False
    
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

# Register the model
ModelRegistry.register("dncnn_denoiser", DnCNNDenoiser)