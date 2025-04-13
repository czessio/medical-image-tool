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
            logger.error(f"Model type not found: {name}")
            return None
        
        # Check if model_path is provided
        if 'model_path' in kwargs and kwargs['model_path']:
            # Fix any duplicated "foundational" directories
            if "foundational/foundational" in kwargs['model_path']:
                kwargs['model_path'] = kwargs['model_path'].replace("foundational/foundational", "foundational")
            
            # Handle paths that don't include the model type subdirectory
            original_path = kwargs['model_path']
            
            # Specifically handle "dncnn_gray_blind.pth" to "dncnn_25.pth" conversion
            if "dncnn_gray_blind.pth" in original_path:
                alternative_path = original_path.replace("dncnn_gray_blind.pth", "dncnn_25.pth")
                if os.path.exists(alternative_path):
                    logger.info(f"Using alternative denoising model: {alternative_path}")
                    kwargs['model_path'] = alternative_path
            
            # Specifically handle "edsr_x2.pt" to "RealESRGAN_x2.pth" conversion
            if "edsr_x2.pt" in original_path:
                alternative_path = original_path.replace("edsr_x2.pt", "RealESRGAN_x2.pth")
                if os.path.exists(alternative_path):
                    logger.info(f"Using alternative super-resolution model: {alternative_path}")
                    kwargs['model_path'] = alternative_path
            
            # If path doesn't exist, try to find it
            if not os.path.exists(kwargs['model_path']):
                # Try to find the model in the correct location
                model_basename = os.path.basename(original_path)
                for model_type in ["denoising", "super_resolution", "artifact_removal"]:
                    # Handle different file extensions and known substitutions
                    if model_type == "denoising" and name == "dncnn_denoiser":
                        alternatives = ["dncnn_25.pth", "dncnn_gray_blind.pth"]
                        for alt in alternatives:
                            alt_path = f"weights/foundational/{model_type}/{alt}"
                            if os.path.exists(alt_path):
                                logger.info(f"Using alternate denoising path: {alt_path}")
                                kwargs['model_path'] = alt_path
                                break
                    
                    elif model_type == "super_resolution" and name == "edsr_super_resolution":
                        alternatives = ["RealESRGAN_x2.pth", "edsr_x2.pt", "RealESRGAN_x4.pth", "RealESRGAN_x8.pth"]
                        for alt in alternatives:
                            alt_path = f"weights/foundational/{model_type}/{alt}" 
                            if os.path.exists(alt_path):
                                logger.info(f"Using alternate super-resolution path: {alt_path}")
                                kwargs['model_path'] = alt_path
                                break
                    
                    elif model_type == "artifact_removal" and name == "unet_artifact_removal":
                        alt_path = f"weights/foundational/{model_type}/G_ema_ep_82.pth"
                        if os.path.exists(alt_path):
                            logger.info(f"Using artifact removal path: {alt_path}")
                            kwargs['model_path'] = alt_path
                            break
        
        try:
            # Print the model path for debugging
            if 'model_path' in kwargs:
                logger.info(f"Creating model with path: {kwargs['model_path']}")
                
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