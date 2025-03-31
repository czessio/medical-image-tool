import sys
from pathlib import Path
import pytest
import os
import tempfile

# Add the root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai.model_registry import ModelRegistry
from ai.base_model import BaseModel

# Create some test model classes
class TestModel1(BaseModel):
    def _load_model(self):
        pass
    
    def preprocess(self, image):
        return image
    
    def inference(self, preprocessed_image):
        return preprocessed_image
    
    def postprocess(self, model_output, original_image=None):
        return model_output

class TestModel2(BaseModel):
    def _load_model(self):
        pass
    
    def preprocess(self, image):
        return image * 2
    
    def inference(self, preprocessed_image):
        return preprocessed_image
    
    def postprocess(self, model_output, original_image=None):
        return model_output / 2

class TestModelRegistry:
    def setup_method(self):
        """Register test models and set up test environment."""
        # Clear registry
        ModelRegistry._registry = {}
        
        # Register test models
        ModelRegistry.register("test_model1", TestModel1)
        ModelRegistry.register("test_model2", TestModel2)
    
    def test_register_and_get(self):
        """Test registering and retrieving models."""
        # Check registered models
        assert "test_model1" in ModelRegistry._registry
        assert "test_model2" in ModelRegistry._registry
        
        # Get models by name
        model_class1 = ModelRegistry.get("test_model1")
        model_class2 = ModelRegistry.get("test_model2")
        
        assert model_class1 == TestModel1
        assert model_class2 == TestModel2
        
        # Try getting non-existent model
        assert ModelRegistry.get("non_existent") is None
    
    def test_create_model(self):
        """Test creating model instances."""
        # Create model instances
        model1 = ModelRegistry.create("test_model1")
        model2 = ModelRegistry.create("test_model2")
        
        assert isinstance(model1, TestModel1)
        assert isinstance(model2, TestModel2)
        
        # Try creating non-existent model
        assert ModelRegistry.create("non_existent") is None
    
    def test_list_available(self):
        """Test listing available model types."""
        available_models = ModelRegistry.list_available()
        assert sorted(available_models) == sorted(["test_model1", "test_model2"])
    
    def test_load_from_directory(self):
        """Test dynamic model loading from directory."""
        # Create a temporary directory with a test module
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a test model module
            module_content = """
from ai.model_registry import ModelRegistry
from ai.base_model import BaseModel

class DynamicTestModel(BaseModel):
    def _load_model(self):
        pass
    
    def preprocess(self, image):
        return image
    
    def inference(self, preprocessed_image):
        return preprocessed_image
    
    def postprocess(self, model_output, original_image=None):
        return model_output

# Register the model
ModelRegistry.register("dynamic_test_model", DynamicTestModel)
"""
            module_path = os.path.join(tmp_dir, "dynamic_model.py")
            with open(module_path, "w") as f:
                f.write(module_content)
            
            # Load models from directory
            count = ModelRegistry.load_models_from_directory(tmp_dir)
            
            # Check if model was loaded
            assert count >= 1
            assert "dynamic_test_model" in ModelRegistry.list_available()