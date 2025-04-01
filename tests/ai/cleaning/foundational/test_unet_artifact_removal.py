"""
Tests for the U-Net-based artifact removal model.
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

from ai.cleaning.models.foundational.unet_artifact_removal import UNetArtifactRemoval
from ai.model_registry import ModelRegistry

class TestUNetArtifactRemoval:
    def setup_method(self):
        """Set up test environment."""
        # Create a test image with artifacts
        self.clean_image = np.zeros((64, 64), dtype=np.float32)
        # Add a circle
        x, y = np.mgrid[0:64, 0:64]
        circle = ((x - 32) ** 2 + (y - 32) ** 2) < 400
        self.clean_image[circle] = 0.8
        
        # Create artifacts (horizontal and vertical lines)
        self.artifact_image = self.clean_image.copy()
        # Horizontal lines
        for i in range(5, 60, 10):
            self.artifact_image[i:i+2, :] = 1.0
        # Vertical lines
        for i in range(10, 60, 15):
            self.artifact_image[:, i:i+1] = 0.0
        # Random noise
        np.random.seed(42)  # For reproducibility
        self.artifact_image += np.random.normal(0, 0.05, size=self.artifact_image.shape).astype(np.float32)
        # Ensure [0, 1] range
        self.artifact_image = np.clip(self.artifact_image, 0.0, 1.0)
    
    def test_model_registration(self):
        """Test that the model is properly registered."""
        model_class = ModelRegistry.get("unet_artifact_removal")
        assert model_class is not None
        assert model_class == UNetArtifactRemoval
    
    def test_model_initialization(self):
        """Test model initialization without weights."""
        model = UNetArtifactRemoval(device='cpu')
        model.initialize()
        assert model.initialized
        assert model.model is not None
        assert model.device == 'cpu'
    
    def test_inference_small(self):
        """Test the full inference pipeline with a minimal model."""
        # Using a small model for testing purposes (fewer channels)
        model = UNetArtifactRemoval(device='cpu', n_channels=8)
        model.initialize()
        
        # Process the artifact image
        result = model.process(self.artifact_image)
        
        # Check shape and type
        assert isinstance(result, np.ndarray)
        assert result.shape == self.artifact_image.shape
        assert np.issubdtype(result.dtype, np.floating)
        
        # Values should be in [0, 1] range
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_bilinear_vs_transpose(self):
        """Test both upsampling options."""
        # Initialize models with and without bilinear upsampling
        model_bilinear = UNetArtifactRemoval(device='cpu', n_channels=8, bilinear=True)
        model_transpose = UNetArtifactRemoval(device='cpu', n_channels=8, bilinear=False)
        
        model_bilinear.initialize()
        model_transpose.initialize()
        
        # Process the artifact image with both models
        result_bilinear = model_bilinear.process(self.artifact_image)
        result_transpose = model_transpose.process(self.artifact_image)
        
        # Both should have the same shape as input
        assert result_bilinear.shape == self.artifact_image.shape
        assert result_transpose.shape == self.artifact_image.shape
        
        # Results should be different (different upsampling methods)
        assert not np.array_equal(result_bilinear, result_transpose)
    
    def test_rgb_input_handling(self):
        """Test handling of RGB input images."""
        # Convert test image to RGB by repeating channels
        rgb_image = np.stack([self.artifact_image] * 3, axis=-1)
        assert rgb_image.shape == (64, 64, 3)
        
        # Initialize small model for testing
        model = UNetArtifactRemoval(device='cpu', n_channels=8)
        model.initialize()
        
        # Process the RGB image
        result = model.process(rgb_image)
        
        # Output should have same spatial dimensions
        assert result.shape[:2] == rgb_image.shape[:2]
        
        # Values should be in [0, 1] range
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_model_structure(self):
        """Test the model structure is created correctly."""
        model = UNetArtifactRemoval(device='cpu')
        model.initialize()
        
        # Check U-Net components
        assert hasattr(model.model, 'inc')
        assert hasattr(model.model, 'down1')
        assert hasattr(model.model, 'down2')
        assert hasattr(model.model, 'down3')
        assert hasattr(model.model, 'down4')
        assert hasattr(model.model, 'up1')
        assert hasattr(model.model, 'up2')
        assert hasattr(model.model, 'up3')
        assert hasattr(model.model, 'up4')
        assert hasattr(model.model, 'outc')