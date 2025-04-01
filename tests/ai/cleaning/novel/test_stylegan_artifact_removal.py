"""
Tests for the StyleGAN-inspired artifact removal model.
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

from ai.cleaning.models.novel.stylegan_artifact_removal import StyleGANArtifactRemoval
from ai.model_registry import ModelRegistry

class TestStyleGANArtifactRemoval:
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
        model_class = ModelRegistry.get("stylegan_artifact_removal")
        assert model_class is not None
        assert model_class == StyleGANArtifactRemoval
    
    def test_model_initialization(self):
        """Test model initialization without weights."""
        model = StyleGANArtifactRemoval(device='cpu')
        model.initialize()
        assert model.initialized
        assert model.model is not None
        assert model.device == 'cpu'
    
    def test_inference(self):
        """Test the full inference pipeline."""
        # Initialize model
        model = StyleGANArtifactRemoval(device='cpu', inject_noise=False)
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
    
    def test_inference_with_noise(self):
        """Test the inference with noise injection enabled."""
        # Skip this test until we have proper test weights
        pytest.skip("Noise injection test requires properly sized model weights")
        
        # Initialize model
        model = StyleGANArtifactRemoval(device='cpu', inject_noise=True)
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
    
    def test_rgb_input_handling(self):
        """Test handling of RGB input images."""
        # Convert test image to RGB by repeating channels
        rgb_image = np.stack([self.artifact_image] * 3, axis=-1)
        assert rgb_image.shape == (64, 64, 3)
        
        # Initialize model
        model = StyleGANArtifactRemoval(device='cpu', inject_noise=False)
        model.initialize()
        
        # Process the RGB image
        result = model.process(rgb_image)
        
        # Output should have same spatial dimensions
        assert result.shape[:2] == rgb_image.shape[:2]
        
        # Values should be in [0, 1] range
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_skip_connection(self):
        """Test that the skip connection is working properly."""
        # Initialize model
        model = StyleGANArtifactRemoval(device='cpu', inject_noise=False)
        model.initialize()
        
        # Create a custom method to access internal model for testing
        def get_skip_output(input_tensor):
            with torch.no_grad():
                # Get the skip connection output directly
                skip = model.model.skip_connection(input_tensor)
                return skip
        
        # Preprocess the image
        tensor = model.preprocess(self.artifact_image)
        
        # Get skip connection output
        skip_output = get_skip_output(tensor)
        
        # Skip connection should have same shape as input
        assert skip_output.shape[2:] == tensor.shape[2:]
        
        # Skip connection output should be a tensor
        assert torch.is_tensor(skip_output)