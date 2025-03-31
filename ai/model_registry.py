"""
Model registry for the medical image enhancement application.
Allows registration and retrieval of model classes by name.
"""
import logging
from pathlib import Path
import importlib.util
import os

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for model classes."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name, model_class):
        """
        Register a model class with the registry.
        
        Args:
            name: Name of the model type
            model_class: Model class to register
        """
        cls._registry[name] = model_class
        logger.debug(f"Registered model type: {name}")
    
    @classmethod
    def get(cls, name):
        """
        Get a model class by name.
        
        Args:
            name: Name of the model type
            
        Returns:
            Model class or None if not found
        """
        if name not in cls._registry:
            logger.error(f"Model type not found: {name}")
            return None
        
        return cls._registry[name]
    
    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Create a model instance by name.
        
        Args:
            name: Name of the model type
            *args, **kwargs: Arguments to pass to the model constructor
            
        Returns:
            Model instance or None if model type not found
        """
        model_class = cls.get(name)
        if model_class is None:
            return None
        
        try:
            return model_class(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error creating model instance: {e}")
            return None
    
    @classmethod
    def list_available(cls):
        """
        List all available model types.
        
        Returns:
            list: Names of all registered model types
        """
        return list(cls._registry.keys())
    
    @classmethod
    def load_models_from_directory(cls, directory):
        """
        Dynamically load and register model implementations from a directory.
        
        Args:
            directory: Directory containing model implementation modules
            
        Returns:
            int: Number of model types loaded
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Model directory not found: {directory}")
            return 0
        
        count = 0
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
                
            try:
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Modules should register their models using ModelRegistry.register
                count += 1
                logger.debug(f"Loaded model module: {module_name}")
            except Exception as e:
                logger.error(f"Error loading model module {file_path}: {e}")
        
        return count