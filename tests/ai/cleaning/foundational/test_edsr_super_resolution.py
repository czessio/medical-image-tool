"""
Tests for the EDSR-based super-resolution model.
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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is not installed"
)

from ai.cleaning.models.foundational.edsr_super_resolution import EDSRSuperResolution
from ai.model_registry import ModelRegistry

class TestEDSRSuperResolution:
    def setup_method(self):
        """Set up test environment."""
        # Create a test low-resolution image
        self.low_res_image = np.zeros((32, 32), dtype=np.float32)
        # Add a pattern
        x, y = np.mgrid[0:32, 0:32]
        circle = ((x - 16) ** 2 + (y - 16) ** 2) < 100
        self.low_res_image[circle] = 0.8
        # Add some gradients
        self.low_res_image += np.linspace(0, 0.2, 32).reshape(-1, 1)
        # Ensure [0, 1] range
        self.low_res_image = np.clip(self.low_res_image, 0.0, 1.0)
    
    def test_model_registration(self):
        """Test that the model is properly registered."""
        model_class = ModelRegistry.get("edsr_super_resolution")
        assert model_class is not None
        assert model_class == EDSRSuperResolution
    
    def test_model_initialization(self):
        """Test model initialization without weights."""
        model = EDSRSuperResolution(device='cpu', scale_factor=2)
        model.initialize()
        assert model.initialized
        assert model.model is not None
        assert model.device == 'cpu'
    
    def test_inference_2x(self):
        """Test the full inference pipeline with 2x upscaling."""
        # Using a small model for testing purposes (fewer residual blocks)
        model = EDSRSuperResolution(device='cpu', scale_factor=2, n_resblocks=2, n_feats=16)
        model.initialize()
        
        # Process the low-res image
        result = model.process(self.low_res_image)
        
        # Check shape (should be 2x larger)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == self.low_res_image.shape[0] * 2
        assert result.shape[1] == self.low_res_image.shape[1] * 2
        assert np.issubdtype(result.dtype, np.floating)
        
        # Values should be in [0, 1] range
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_inference_3x(self):
        """Test the full inference pipeline with 3x upscaling."""
        # Using a small model for testing purposes (fewer residual blocks)
        model = EDSRSuperResolution(device='cpu', scale_factor=3, n_resblocks=2, n_feats=16)
        model.initialize()
        
        # Process the low-res image
        result = model.process(self.low_res_image)
        
        # Check shape (should be 3x larger)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == self.low_res_image.shape[0] * 3
        assert result.shape[1] == self.low_res_image.shape[1] * 3
        assert np.issubdtype(result.dtype, np.floating)
        
        # Values should be in [0, 1] range
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_rgb_input_handling(self):
        """Test handling of RGB input images."""
        # Convert test image to RGB by repeating channels
        rgb_image = np.stack([self.low_res_image] * 3, axis=-1)
        assert rgb_image.shape == (32, 32, 3)
        
        # Initialize small model for testing
        model = EDSRSuperResolution(device='cpu', scale_factor=2, n_resblocks=2, n_feats=16)
        model.initialize()
        
        # Process the RGB image
        result = model.process(rgb_image)
        
        # Output should have 2x spatial dimensions
        assert result.shape[0] == rgb_image.shape[0] * 2
        assert result.shape[1] == rgb_image.shape[1] * 2
        
        # Values should be in [0, 1] range
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_model_components(self):
        """Test the model components are created correctly."""
        model = EDSRSuperResolution(device='cpu', scale_factor=2, n_resblocks=4, n_feats=32)
        model.initialize()
        
        # Check main components
        assert hasattr(model.model, 'head')
        assert hasattr(model.model, 'body')
        assert hasattr(model.model, 'tail')
        
        # Check body has correct number of residual blocks (plus final conv)
        assert len(model.model.body) == model.n_resblocks + 1