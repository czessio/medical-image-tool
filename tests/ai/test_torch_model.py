import sys
from pathlib import Path
import pytest
import numpy as np
import os
import tempfile

# Add the root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import conditionally for PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.torch_model import TorchModel

# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is not installed"
)

# Create a simple PyTorch model for testing
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class TestTorchModel(TorchModel):
    def __init__(self, model_path=None, device=None):
        super().__init__(model_path, device)
        # Explicitly set torch_device here for testing
        self.torch_device = torch.device(self.device)
        
    def _create_model_architecture(self):
        model = SimpleNetwork()
        # Ensure the model is on the same device
        model = model.to(self.torch_device)
        return model

class TestTorchModelClass:
    def setup_method(self):
        """Create test data and a temporary model file."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch is not installed")
        
        # Create a test RGB image
        self.test_image = np.ones((20, 30, 3), dtype=np.float32) * 0.5
        
        # Create a temporary model file
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
        
        # Create and save a simple model
        model = SimpleNetwork()
        torch.save(model.state_dict(), self.model_path)
    
    def teardown_method(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir'):
            for file in Path(self.temp_dir).glob("*"):
                try:
                    os.remove(file)
                except:
                    pass
            os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization with a valid model file."""
        model = TestTorchModel(model_path=self.model_path)
        assert model.initialized == True
        assert model.torch_device is not None
    
    def test_preprocess(self):
        """Test preprocessing logic for PyTorch."""
        model = TestTorchModel(model_path=self.model_path)
        
        # Preprocess the test image
        tensor = model.preprocess(self.test_image)
        
        # Check tensor properties
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 20, 30)  # [B, C, H, W]
        assert tensor.device == model.torch_device
    
    def test_inference_and_postprocess(self):
        """Test inference and postprocessing."""
        model = TestTorchModel(model_path=self.model_path)
        
        # Make sure the model and weights are on the same device
        if not hasattr(model, 'model') or model.model is None:
            model._load_model()
        
        # Force everything to CPU to avoid device mismatch
        model.device = 'cpu'
        model.torch_device = torch.device('cpu')
        model.model = model.model.to('cpu')
        
        # Run through the pipeline
        preprocessed = model.preprocess(self.test_image)
        output = model.inference(preprocessed)
        result = model.postprocess(output)
        
        # Check result properties
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 30, 3)
        assert 0 <= result.min() <= result.max() <= 1.0