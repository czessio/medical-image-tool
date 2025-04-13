"""
Updated cleaning pipeline for medical image enhancement application.
Manages the execution of AI cleaning models in sequence with improved model loading.
"""
import logging
from pathlib import Path
import numpy as np
import torch
import os

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
    
    def __init__(self, use_novel_models=None, config=None):
        """
        Initialize the cleaning pipeline.
        
        Args:
            use_novel_models: Whether to use novel (True) or foundational (False) models. If None, use config setting.
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
        
        # If use_novel_models is explicitly provided, use it. Otherwise, read from config.
        if use_novel_models is None:
            self.use_novel_models = self.config.get("models.use_novel", False)
        else:
            self.use_novel_models = use_novel_models
        
        # Initialize with default models if available
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize pipeline with default models based on configuration."""
        logger.info(f"Initializing pipeline with {'novel' if self.use_novel_models else 'foundational'} models")
        
        # Try to load all models, but continue even if some fail
        success = True
        
        # Load denoising model
        try:
            if self.use_novel_models:
                denoising_success = self.set_denoising_model("novel_diffusion_denoiser")
            else:
                denoising_success = self.set_denoising_model("dncnn_denoiser")
            
            if not denoising_success:
                logger.warning("Denoising model failed to load, continuing without it")
        except Exception as e:
            logger.error(f"Error loading denoising model: {e}")
            denoising_success = False
        
        # Load super-resolution model
        try:
            # Get scale factor from config
            scale_factor = self.config.get("models.super_resolution.scale_factor", 2)
            
            if self.use_novel_models:
                sr_success = self.set_super_resolution_model("novel_restormer", scale_factor=scale_factor)
            else:
                sr_success = self.set_super_resolution_model("edsr_super_resolution", scale_factor=scale_factor)
            
            if not sr_success:
                logger.warning("Super-resolution model failed to load, continuing without it")
        except Exception as e:
            logger.error(f"Error loading super-resolution model: {e}")
            sr_success = False
        
        # Load artifact removal model
        try:
            if self.use_novel_models:
                ar_success = self.set_artifact_removal_model("novel_stylegan_artifact_removal")
            else:
                ar_success = self.set_artifact_removal_model("unet_artifact_removal")
            
            if not ar_success:
                logger.warning("Artifact removal model failed to load, continuing without it")
        except Exception as e:
            logger.error(f"Error loading artifact removal model: {e}")
            ar_success = False
        
        # Return overall success status
        success = denoising_success or sr_success or ar_success
        if not success:
            logger.warning("No models were loaded successfully")
        else:
            logger.info(f"Loaded models: " + 
                    (f"denoising ({'✓' if denoising_success else '✗'}), " if denoising_success is not None else "") +
                    (f"super-resolution ({'✓' if sr_success else '✗'}), " if sr_success is not None else "") +
                    (f"artifact removal ({'✓' if ar_success else '✗'})" if ar_success is not None else ""))
        
        return success
    
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
            
            # Fix path if it has duplicated "foundational"
            if model_path and "foundational/foundational" in model_path:
                model_path = model_path.replace("foundational/foundational", "foundational")
                
            # Check for specific model names
            if model_type == "unet_artifact_removal":
                # Try standard paths
                standard_paths = [
                    f"weights/foundational/artifact_removal/G_ema_ep_82.pth",
                    f"weights/foundational/foundational/artifact_removal/G_ema_ep_82.pth"
                ]
                for path in standard_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
        
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
            
            # Fix path if it has duplicated "foundational"
            if model_path and "foundational/foundational" in model_path:
                model_path = model_path.replace("foundational/foundational", "foundational")
                
            # Check for specific model names
            if model_type == "dncnn_denoiser":
                # Try standard paths
                standard_paths = [
                    f"weights/foundational/denoising/dncnn_25.pth",
                    f"weights/foundational/denoising/dncnn_gray_blind.pth",
                    f"weights/foundational/foundational/denoising/dncnn_25.pth",
                    f"weights/foundational/foundational/denoising/dncnn_gray_blind.pth"
                ]
                for path in standard_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
        
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
        
        # Get scale factor from config if not provided
        if scale_factor is None:
            scale_factor = self.config.get("models.super_resolution.scale_factor", 2)
        
        # Get model path from config if not provided
        if model_path is None:
            model_category = "novel" if self.use_novel_models else "foundational"
            
            # Choose path based on scale factor for RealESRGAN models
            if model_type == "edsr_super_resolution" and model_category == "foundational":
                if scale_factor == 2:
                    model_path = f"weights/{model_category}/super_resolution/RealESRGAN_x2.pth"
                elif scale_factor == 4:
                    model_path = f"weights/{model_category}/super_resolution/RealESRGAN_x4.pth"
                elif scale_factor == 8:
                    model_path = f"weights/{model_category}/super_resolution/RealESRGAN_x8.pth"
                else:
                    # Default to x2 if unknown scale factor
                    model_path = f"weights/{model_category}/super_resolution/RealESRGAN_x2.pth"
                    scale_factor = 2
                    logger.warning(f"Unknown scale factor {scale_factor}, defaulting to 2")
            else:
                # For other model types, use the config path
                config_path = f"models.super_resolution.{model_category}.{model_type}.model_path"
                model_path = self.config.get(config_path)
            
            # Fix path if it has duplicated "foundational"
            if model_path and "foundational/foundational" in model_path:
                model_path = model_path.replace("foundational/foundational", "foundational")
                
            # Verify the model file exists
            if model_path and not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                # Try to find a valid model file
                if model_type == "edsr_super_resolution":
                    for x in [2, 4, 8]:
                        alt_path = f"weights/{model_category}/super_resolution/RealESRGAN_x{x}.pth"
                        if os.path.exists(alt_path):
                            model_path = alt_path
                            scale_factor = x
                            logger.info(f"Using alternative model file: {model_path}")
                            break
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.super_resolution.device", "auto")
        
        # Create model with specific parameters
        try:
            # Create model with scale factor
            model = ModelRegistry.create(
                model_type, 
                model_path=model_path, 
                device=device, 
                scale_factor=scale_factor
            )
            
            if model is None:
                logger.error(f"Failed to create super-resolution model: {model_type}")
                return False
            
            # Set scale factor on model instance if it has that attribute
            if hasattr(model, "scale_factor"):
                model.scale_factor = scale_factor
                logger.info(f"Set super-resolution scale factor to {scale_factor}")
            
            # Add to pipeline
            self.pipeline.add_model(model, f"super_resolution_{model_type}_x{scale_factor}")
            self.current_models["super_resolution"] = model
            return True
        except Exception as e:
            logger.error(f"Error setting super-resolution model: {e}")
            return False
    
    def set_scale_factor(self, scale_factor):
        """
        Set the scale factor for super-resolution.
        
        Args:
            scale_factor: Upscaling factor (2, 4, or 8)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate scale factor
        if scale_factor not in [2, 4, 8]:
            logger.error(f"Invalid scale factor: {scale_factor}. Must be 2, 4, or 8.")
            return False
        
        logger.info(f"Setting super-resolution scale factor to {scale_factor}")
        
        # Update config
        self.config.set("models.super_resolution.scale_factor", scale_factor)
        
        # If we have a super-resolution model, reload it with the new scale factor
        if self.current_models["super_resolution"] is not None:
            model_name = None
            for name in self.pipeline.model_names:
                if "super_resolution" in name:
                    # Extract model type from name like "super_resolution_edsr_super_resolution_x2"
                    parts = name.split("_")
                    if len(parts) >= 3:
                        # Skip "super_resolution" prefix, keep the model type name
                        model_name = "_".join(parts[1:-1])
                        if model_name.endswith("_x2") or model_name.endswith("_x4") or model_name.endswith("_x8"):
                            # Remove the scale factor suffix if it's part of the model name
                            model_name = model_name[:-3]
                    break
            
            if model_name:
                # Update the model with the new scale factor
                return self.set_super_resolution_model(model_name, scale_factor=scale_factor)
        
        return True
    
    def toggle_model_type(self):
        """
        Toggle between novel and foundational models.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Store current setting
        previous_setting = self.use_novel_models
        
        # Switch setting
        self.use_novel_models = not self.use_novel_models
        logger.info(f"Switched to {'novel' if self.use_novel_models else 'foundational'} models")
        
        # Try to initialize with new setting
        success = self._initialize_models()
        
        # If initialization failed and no models were loaded, revert to previous setting
        if not success and not any(self.current_models.values()):
            logger.warning(f"Failed to load any {'novel' if self.use_novel_models else 'foundational'} models")
            logger.info(f"Reverting to {'novel' if previous_setting else 'foundational'} models")
            self.use_novel_models = previous_setting
            self._initialize_models()
            return False
    
        return success
    
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
        
        # Try to use available models but handle failures gracefully
        try:
            # Get active models
            active_models = [name for name, model in self.current_models.items() if model is not None]
            logger.info(f"Processing image with active models: {', '.join(active_models)}")
            
            result = self.pipeline.process(image)
            return result
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {e}")
            logger.warning("Falling back to original image due to pipeline error")
            return image
    
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