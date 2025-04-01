"""
Tests for the diffusion-based denoiser model.
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

from ai.cleaning.models.novel.diffusion_denoiser import DiffusionDenoiser
from ai.model_registry import ModelRegistry

class TestDiffusionDenoiser:
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
        model_class = ModelRegistry.get("diffusion_denoiser")
        assert model_class is not None
        assert model_class == DiffusionDenoiser
    
    def test_model_initialization(self):
        """Test model initialization without weights."""
        model = DiffusionDenoiser(device='cpu')
        model.initialize()
        assert model.initialized
        assert model.model is not None
        assert model.device == 'cpu'
    
    def test_inference(self):
        """Test the full inference pipeline."""
        # Skip this test until we have proper test weights
        pytest.skip("Full inference test requires properly sized model weights")
        
        # Initialize model
        model = DiffusionDenoiser(device='cpu', inference_steps=2)  # Use minimal steps for testing
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
        # Skip this test until we have proper test weights
        pytest.skip("RGB input handling test requires properly sized model weights")
        
        # Convert test image to RGB by repeating channels
        rgb_image = np.stack([self.noisy_image] * 3, axis=-1)
        assert rgb_image.shape == (64, 64, 3)
        
        # Initialize model
        model = DiffusionDenoiser(device='cpu', inference_steps=2)
        model.initialize()
        
        # Process the RGB image
        result = model.process(rgb_image)
        
        # Output should be grayscale (same spatial dimensions)
        assert result.shape[:2] == rgb_image.shape[:2]
        
        # Values should be in [0, 1] range
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_noise_levels_initialization(self):
        """Test that noise levels are properly initialized."""
        model = DiffusionDenoiser(device='cpu', inference_steps=10)
        model.initialize()
        
        # Check noise levels
        assert model.noise_levels is not None
        assert len(model.noise_levels) == 10  # Should match inference_steps
        assert torch.is_tensor(model.noise_levels)
        assert model.noise_levels.device.type == 'cpu'