import sys
from pathlib import Path
import pytest
import numpy as np

# Add the root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai.base_model import BaseModel
from ai.inference_pipeline import InferencePipeline
from ai.model_registry import ModelRegistry

# Create test models with different operations
class AddOneModel(BaseModel):
    def __init__(self, model_path=None, device=None):
        super().__init__(model_path, device)
        self.initialized = True
    
    def _load_model(self):
        pass
    
    def preprocess(self, image):
        return image
    
    def inference(self, preprocessed_image):
        return preprocessed_image + 1.0
    
    def postprocess(self, model_output, original_image=None):
        return model_output

class MultiplyTwoModel(BaseModel):
    def __init__(self, model_path=None, device=None):
        super().__init__(model_path, device)
        self.initialized = True
    
    def _load_model(self):
        pass
    
    def preprocess(self, image):
        return image
    
    def inference(self, preprocessed_image):
        return preprocessed_image * 2.0
    
    def postprocess(self, model_output, original_image=None):
        return model_output

class TestInferencePipeline:
    def setup_method(self):
        """Set up test environment."""
        # Register test models
        ModelRegistry._registry = {}
        ModelRegistry.register("add_one", AddOneModel)
        ModelRegistry.register("multiply_two", MultiplyTwoModel)
        
        # Create a test image
        self.test_image = np.ones((10, 10), dtype=np.float32) * 0.5
        
        # Create pipeline
        self.pipeline = InferencePipeline()
    
    def test_empty_pipeline(self):
        """Test pipeline with no models."""
        # Empty pipeline should return original image
        result = self.pipeline.process(self.test_image)
        assert np.array_equal(result, self.test_image)
    
    def test_add_model_instance(self):
        """Test adding a model instance to the pipeline."""
        # Add a model instance
        model = AddOneModel()
        self.pipeline.add_model(model, name="add_one_model")
        
        # Process the image
        result = self.pipeline.process(self.test_image)
        
        # Expected: image + 1
        expected = self.test_image + 1.0
        assert np.array_equal(result, expected)
    
    def test_add_model_by_type(self):
        """Test adding a model by type and configuration."""
        # Add model by type (would normally have a path)
        self.pipeline.add_model(("add_one", None, "cpu"))
        
        # Process the image
        result = self.pipeline.process(self.test_image)
        
        # Expected: image + 1
        expected = self.test_image + 1.0
        assert np.array_equal(result, expected)
    
    def test_multiple_models(self):
        """Test pipeline with multiple models."""
        # Add two models
        self.pipeline.add_model(AddOneModel(), "add_one")
        self.pipeline.add_model(MultiplyTwoModel(), "multiply_two")
        
        # Process the image
        result = self.pipeline.process(self.test_image)
        
        # Expected: (image + 1) * 2
        expected = (self.test_image + 1.0) * 2.0
        assert np.array_equal(result, expected)
    
    def test_clear_pipeline(self):
        """Test clearing the pipeline."""
        # Add a model
        self.pipeline.add_model(AddOneModel())
        
        # Clear the pipeline
        self.pipeline.clear()
        
        # Pipeline should be empty
        assert len(self.pipeline.models) == 0
        
        # Processing should return original image
        result = self.pipeline.process(self.test_image)
        assert np.array_equal(result, self.test_image)
    
    def test_callable_interface(self):
        """Test calling the pipeline directly."""
        # Add a model
        self.pipeline.add_model(AddOneModel())
        
        # Call the pipeline directly
        result = self.pipeline(self.test_image)
        
        # Expected: image + 1
        expected = self.test_image + 1.0
        assert np.array_equal(result, expected)
    
    def test_error_handling(self):
        """Test handling of errors in the pipeline."""
        # Create a model that raises an exception
        class ErrorModel(BaseModel):
            def _load_model(self):
                pass
            
            def preprocess(self, image):
                return image
            
            def inference(self, preprocessed_image):
                raise RuntimeError("Test error")
            
            def postprocess(self, model_output, original_image=None):
                return model_output
        
        # Add models - one good, one with error, one good
        self.pipeline.add_model(AddOneModel(), "add_one")
        self.pipeline.add_model(ErrorModel(), "error_model")
        self.pipeline.add_model(MultiplyTwoModel(), "multiply_two")
        
        # Process should continue despite the error
        result = self.pipeline.process(self.test_image)
        
        # Expected: image + 1, then multiply by 2 (skipping the error model)
        expected = (self.test_image + 1.0) * 2.0
        assert np.array_equal(result, expected)