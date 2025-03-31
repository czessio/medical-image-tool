import sys
from pathlib import Path
import pytest
import numpy as np

# Add the root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai.base_model import BaseModel

# Create a minimal concrete implementation for testing
class TestModel(BaseModel):
    def __init__(self, model_path=None, device=None):
        super().__init__(model_path, device)
        # Remove this line - the test expects initialized to be False initially
        # self.initialized = True
    
    def _load_model(self):
        self.model = "test_model"
    
    def preprocess(self, image):
        return image * 2.0
    
    def inference(self, preprocessed_image):
        return preprocessed_image + 1.0
    
    def postprocess(self, model_output, original_image=None):
        return model_output / 2.0

class TestBaseModel:
    def setup_method(self):
        # Create a test image
        self.test_image = np.ones((10, 10), dtype=np.float32) * 0.5
    
    def test_initialization(self):
        """Test model initialization without a path."""
        model = TestModel()
        assert model.initialized == False
        assert model.device in ['cpu', 'cuda']
    
    def test_device_selection(self):
        """Test device selection logic."""
        # Force CPU
        model = TestModel(device='cpu')
        assert model.device == 'cpu'
        
        # Auto should return a valid device
        model = TestModel(device='auto')
        assert model.device in ['cpu', 'cuda']
    
    def test_process_pipeline(self):
        """Test the full process pipeline."""
        model = TestModel()
        model.initialize()
        
        # Process the test image
        result = model.process(self.test_image)
        
        # Expected result: ((image * 2) + 1) / 2 = image + 0.5
        expected = self.test_image + 0.5
        assert np.allclose(result, expected)
    
    def test_callable_interface(self):
        """Test calling the model directly."""
        model = TestModel()
        model.initialize()
        
        # Call the model directly
        result = model(self.test_image)
        
        # Should be same as process
        expected = self.test_image + 0.5
        assert np.allclose(result, expected)