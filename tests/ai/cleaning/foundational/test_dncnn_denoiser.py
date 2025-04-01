"""
Tests for the DnCNN-based denoiser model.
"""
import sys
import os
from pathlib import Path
import pytest
import numpy as np

# Add the root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is not installed"
)

from ai.cleaning.models.foundational.dncnn_denoiser import DnCNNDenoiser
from ai.model_registry import ModelRegistry

class TestDnCNNDenoiser:
    def setup_method(self):
        """Set up test environment."""
        # Create a test noisy image
        np.random.seed(42)  # For reproducibility
        self.clean_image = np.ones((64, 64), dtype=np.float32) * 0.5
        # Add a square in the center
        self.clean_image[20:40, 20:40] = 0.8
        # Add noise
        self.noisy_image = self.clean_image + np.random.normal(0, 0.1, size=self.clean_image.shape).astype(np.float32)
        # Clip to [0, 1] range
        self.noisy_image = np.clip(self.noisy_image, 0.0, 1.0)
    
    def test_model_registration(self):
        """Test that the model is properly registered."""
        model_class = ModelRegistry.get("dncnn_denoiser")
        assert model_class is not None
        assert model_class == DnCNNDenoiser
    
    def test_model_initialization(self):
        """Test model initialization without weights."""
        model = DnCNNDenoiser(device='cpu')
        model.initialize()
        assert model.initialized
        assert model.model is not None
        assert model.device == 'cpu'
    
    def test_inference(self):
        """Test the full inference pipeline."""
        # Using a small model for testing purposes (fewer layers)
        model = DnCNNDenoiser(device='cpu', num_layers=3)
        model.initialize()
        
        # Process the noisy image
        result = model.process(self.noisy_image)
        
        # Check shape and type
        assert isinstance(result, np.ndarray)
        assert result.shape == self.noisy_image.shape
        assert np.issubdtype(result.dtype, np.floating)
        
        # Values should be in [0, 1] range
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_rgb_input_handling(self):
        """Test handling of RGB input images."""
        # Convert test image to RGB by repeating channels
        rgb_image = np.stack([self.noisy_image] * 3, axis=-1)
        assert rgb_image.shape == (64, 64, 3)
        
        # Initialize model with fewer layers for testing
        model = DnCNNDenoiser(device='cpu', num_layers=3)
        model.initialize()
        
        # Process the RGB image
        result = model.process(rgb_image)
        
        # Output should be grayscale (same spatial dimensions)
        assert result.shape[:2] == rgb_image.shape[:2]
        
        # Values should be in [0, 1] range
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_model_architecture(self):
        """Test the model architecture creation."""
        model = DnCNNDenoiser(device='cpu', num_layers=17)
        model.initialize()
        
        # Check model structure
        assert isinstance(model.model.first_layer, nn.Sequential)
        assert len(model.model.middle_layers) == 15  # num_layers - 2
        assert isinstance(model.model.last_layer, nn.Conv2d)