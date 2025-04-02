"""
Cleaning pipeline for medical image enhancement application.
Manages the execution of AI cleaning models in sequence.
"""
import logging
from pathlib import Path
import numpy as np
import torch

from ai.inference_pipeline import InferencePipeline
from ai.model_registry import ModelRegistry
from utils.config import Config

logger = logging.getLogger(__name__)

class CleaningPipeline:
    """
    Pipeline for cleaning medical images using AI models.
    
    This pipeline manages the execution of multiple cleaning models (denoising,
    super-resolution, artifact removal) in sequence. It supports switching between
    novel (cutting-edge) models and foundational (established) models for comparison.
    """
    
    def __init__(self, use_novel_models=True, config=None):
        """
        Initialize the cleaning pipeline.
        
        Args:
            use_novel_models: Whether to use novel (True) or foundational (False) models
            config: Configuration object or None to use default config
        """
        self.pipeline = InferencePipeline()
        self.current_models = {
            "denoising": None,
            "super_resolution": None,
            "artifact_removal": None
        }
        
        # Configuration for models
        self.config = config or Config()
        self.use_novel_models = use_novel_models
        
        # Initialize with default models if available
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize pipeline with default models based on configuration."""
        # Load available model types
        if self.use_novel_models:
            logger.info("Initializing pipeline with novel models")
            self.set_denoising_model("novel_diffusion_denoiser")
            self.set_super_resolution_model("novel_restormer")
            self.set_artifact_removal_model("novel_stylegan_artifact_removal")
        else:
            logger.info("Initializing pipeline with foundational models")
            # These will be implemented in the foundational models folder
            self.set_denoising_model("dncnn_denoiser")
            self.set_super_resolution_model("edsr")
            self.set_artifact_removal_model("unet_artifact_removal")
    
    def set_denoising_model(self, model_type, model_path=None, device=None):
        """
        Set the denoising model for the pipeline.
        
        Args:
            model_type: Type of denoising model to use
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Setting denoising model: {model_type}")
        
        # Remove existing model if present
        if self.current_models["denoising"] is not None:
            idx = self.pipeline.models.index(self.current_models["denoising"])
            self.pipeline.models.pop(idx)
            self.pipeline.model_names.pop(idx)
            self.current_models["denoising"] = None
        
        # Get model path from config if not provided
        if model_path is None:
            model_category = "novel" if self.use_novel_models else "foundational"
            config_path = f"models.denoising.{model_category}.{model_type}.model_path"
            model_path = self.config.get(config_path)
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.denoising.device", "auto")
        
        # Create and add model
        model = ModelRegistry.create(model_type, model_path=model_path, device=device)
        if model is None:
            logger.error(f"Failed to create denoising model: {model_type}")
            return False
        
        self.pipeline.add_model(model, f"denoising_{model_type}")
        self.current_models["denoising"] = model
        return True
    
    def set_super_resolution_model(self, model_type, model_path=None, device=None, scale_factor=None):
        """
        Set the super-resolution model for the pipeline.
        
        Args:
            model_type: Type of super-resolution model to use
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            scale_factor: Upscaling factor (None to use default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Setting super-resolution model: {model_type}")
        
        # Remove existing model if present
        if self.current_models["super_resolution"] is not None:
            idx = self.pipeline.models.index(self.current_models["super_resolution"])
            self.pipeline.models.pop(idx)
            self.pipeline.model_names.pop(idx)
            self.current_models["super_resolution"] = None
        
        # Get model path from config if not provided
        if model_path is None:
            model_category = "novel" if self.use_novel_models else "foundational"
            config_path = f"models.super_resolution.{model_category}.{model_type}.model_path"
            model_path = self.config.get(config_path)
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.super_resolution.device", "auto")
        
        # Get scale factor from config if not provided
        if scale_factor is None:
            scale_factor = self.config.get("models.super_resolution.scale_factor", 2)
        
        # Create and add model
        model = ModelRegistry.create(model_type, model_path=model_path, device=device)
        if model is None:
            logger.error(f"Failed to create super-resolution model: {model_type}")
            return False
        
        # Set scale factor if model supports it
        if hasattr(model, "scale_factor"):
            model.scale_factor = scale_factor
        
        self.pipeline.add_model(model, f"super_resolution_{model_type}")
        self.current_models["super_resolution"] = model
        return True
    
    def set_artifact_removal_model(self, model_type, model_path=None, device=None):
        """
        Set the artifact removal model for the pipeline.
        
        Args:
            model_type: Type of artifact removal model to use
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Setting artifact removal model: {model_type}")
        
        # Remove existing model if present
        if self.current_models["artifact_removal"] is not None:
            idx = self.pipeline.models.index(self.current_models["artifact_removal"])
            self.pipeline.models.pop(idx)
            self.pipeline.model_names.pop(idx)
            self.current_models["artifact_removal"] = None
        
        # Get model path from config if not provided
        if model_path is None:
            model_category = "novel" if self.use_novel_models else "foundational"
            config_path = f"models.artifact_removal.{model_category}.{model_type}.model_path"
            model_path = self.config.get(config_path)
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.artifact_removal.device", "auto")
        
        # Create and add model
        model = ModelRegistry.create(model_type, model_path=model_path, device=device)
        if model is None:
            logger.error(f"Failed to create artifact removal model: {model_type}")
            return False
        
        self.pipeline.add_model(model, f"artifact_removal_{model_type}")
        self.current_models["artifact_removal"] = model
        return True
    
    def toggle_model_type(self):
        """
        Toggle between novel and foundational models.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.use_novel_models = not self.use_novel_models
        logger.info(f"Switched to {'novel' if self.use_novel_models else 'foundational'} models")
        return self._initialize_models()
    
    def enable_all_models(self):
        """
        Enable all cleaning models in the pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        return self._initialize_models()
    
    def disable_all_models(self):
        """
        Disable all models in the pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.pipeline.clear()
        self.current_models = {
            "denoising": None,
            "super_resolution": None,
            "artifact_removal": None
        }
        return True
    
    def process(self, image):
        """
        Process an image through the cleaning pipeline.
        
        Args:
            image: Input medical image as numpy array
            
        Returns:
            numpy.ndarray: Cleaned image
        """
        if not any(self.current_models.values()):
            logger.warning("No models in pipeline, returning original image")
            return image
        
        return self.pipeline.process(image)
    
    def get_active_models(self):
        """
        Get list of active models in the pipeline.
        
        Returns:
            list: Names of active models
        """
        return [name for name, model in self.current_models.items() if model is not None]
    
    def __call__(self, image):
        """
        Make the cleaning pipeline callable directly.
        
        Args:
            image: Input medical image as numpy array
            
        Returns:
            numpy.ndarray: Cleaned image
        """
        return self.process(image)