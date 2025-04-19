"""
Updated cleaning pipeline for medical image enhancement application.
Manages the execution of AI cleaning models in sequence with improved model loading and error handling.
"""
import logging
from pathlib import Path
import numpy as np
import torch
import os

from ai.inference_pipeline import InferencePipeline
from ai.model_registry import ModelRegistry
from ai.model_adapter import ModelAdapter  # Import the new ModelAdapter
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
            "artifact_removal": None,
            "enhancement": None
        }
        
        # Configuration for models
        self.config = config or Config()
        
        # Create model service for pipeline
        from utils.model_service import ModelService
        self.model_service = ModelService(self.config)
        
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
        
        # Create ModelService if not provided
        if not hasattr(self, 'model_service'):
            from utils.model_service import ModelService
            self.model_service = ModelService(self.config)
        
        # Try to load all models, but continue even if some fail
        success = True
        
        # Load denoising model
        try:
            if self.use_novel_models:
                # Try novel models in order of preference
                denoising_models = [
                    "novel_vit_mae_cxr", 
                    "novel_resnet50_rad", 
                    "novel_resnet50_medical", 
                    "novel_swinvit", 
                    "novel_diffusion_denoiser"
                ]
                
                denoising_success = False
                for model_id in denoising_models:
                    if self.model_service.check_model_availability(model_id):
                        if model_id in ["novel_vit_mae_cxr", "novel_resnet50_rad", "novel_resnet50_medical", "novel_swinvit"]:
                            denoising_success = self._set_enhancement_model(model_id, task_type='enhancement')
                        else:
                            denoising_success = self.set_denoising_model(model_id)
                        
                        if denoising_success:
                            break
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
                sr_models = ["novel_restormer", "novel_swinir_super_resolution"]
                sr_success = False
                
                for model_id in sr_models:
                    if self.model_service.check_model_availability(model_id):
                        sr_success = self.set_super_resolution_model(model_id, scale_factor=scale_factor)
                        if sr_success:
                            break
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
            # Using ASCII symbols instead of Unicode for better compatibility
            logger.info(f"Loaded models: " + 
                    (f"denoising (Y), " if denoising_success else "") +
                    (f"super-resolution (Y), " if sr_success else "") +
                    (f"artifact removal (Y)" if ar_success else ""))
        
        # Ensure all models are on the same device
        if success:
            self._ensure_models_same_device()
            
        return success
    
    def _ensure_models_same_device(self):
        """
        Ensure all models in the pipeline are using the same device.
        This prevents issues with tensor device mismatches during inference.
        """
        # First, determine what device should be used
        target_device = None
        
        # Check if any models are initialized and use its device
        for model in self.current_models.values():
            if model is not None and hasattr(model, 'device'):
                target_device = model.device
                break
        
        # If no device found, default to 'cpu'
        if target_device is None:
            target_device = 'cpu'
            
        logger.info(f"Ensuring all models use device: {target_device}")
        
        # Now make sure all models use the same device
        for model_type, model in self.current_models.items():
            if model is not None and hasattr(model, 'device') and model.device != target_device:
                try:
                    logger.info(f"Moving {model_type} from {model.device} to {target_device}")
                    model.device = target_device
                    
                    # If the model has a torch_device attribute, update it too
                    if hasattr(model, 'torch_device'):
                        model.torch_device = torch.device(target_device)
                        
                        # If the model has an internal model, move it to the target device
                        if hasattr(model, 'model') and hasattr(model.model, 'to'):
                            model.model.to(model.torch_device)
                except Exception as e:
                    logger.error(f"Error moving {model_type} to {target_device}: {e}")
    
    def set_artifact_removal_model(self, model_id, model_path=None, device=None):
        """
        Set the artifact removal model for the pipeline.
        
        Args:
            model_id: ID of the artifact removal model to use
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Setting artifact removal model: {model_id}")
        
        # Remove existing model if present
        if self.current_models["artifact_removal"] is not None:
            idx = self.pipeline.models.index(self.current_models["artifact_removal"])
            self.pipeline.models.pop(idx)
            self.pipeline.model_names.pop(idx)
            self.current_models["artifact_removal"] = None
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.artifact_removal.device", "auto")
        
        # Create model service if not available
        if not hasattr(self, 'model_service'):
            from utils.model_service import ModelService
            self.model_service = ModelService(self.config)
        
        # Get the model from the service
        kwargs = {"device": device}
        if model_path:
            kwargs["model_path"] = model_path
            
        model = self.model_service.get_model(model_id, **kwargs)
        
        if model is None:
            logger.error(f"Failed to create artifact removal model: {model_id}")
            return False
        
        # Wrap model in adapter for safety
        from ai.model_adapter import ModelAdapter
        model_adapter = ModelAdapter(model, f"artifact_removal_{model_id}")
        
        self.pipeline.add_model(model_adapter, f"artifact_removal_{model_id}")
        self.current_models["artifact_removal"] = model_adapter
        return True
    
    
    
    
    def set_motion_artifact_model(self, model_id, model_path=None, device=None):
        """
        Set the motion artifact removal model for the pipeline.
        
        Args:
            model_id: ID of the motion artifact removal model to use
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
                
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Setting motion artifact removal model: {model_id}")
        
        # Remove existing model if present
        if self.current_models.get("motion_artifact") is not None:
            idx = self.pipeline.models.index(self.current_models["motion_artifact"])
            self.pipeline.models.pop(idx)
            self.pipeline.model_names.pop(idx)
            self.current_models["motion_artifact"] = None
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.motion_artifact.device", "auto")
        
        # Create model service if not available
        if not hasattr(self, 'model_service'):
            from utils.model_service import ModelService
            self.model_service = ModelService(self.config)
        
        # For motion artifacts, we can reuse the same artifact removal model
        # but with specialized parameters
        kwargs = {
            "device": device,
            "focus_on_motion": True  # Tell the model to focus on motion artifacts
        }
        
        if model_path:
            kwargs["model_path"] = model_path
            
        model = self.model_service.get_model(model_id, **kwargs)
        
        if model is None:
            logger.error(f"Failed to create motion artifact removal model: {model_id}")
            return False
        
        # Wrap model in adapter for safety
        from ai.model_adapter import ModelAdapter
        model_adapter = ModelAdapter(model, f"motion_artifact_{model_id}")
        
        self.pipeline.add_model(model_adapter, f"motion_artifact_{model_id}")
        self.current_models["motion_artifact"] = model_adapter
        return True
    
    
    
    def set_segmentation_model(self, model_id, model_path=None, device=None, num_classes=2):
        """
        Set the segmentation model for the pipeline.
        
        Args:
            model_id: ID of the segmentation model to use
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            num_classes: Number of segmentation classes
                
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Setting segmentation model: {model_id}")
        
        # Remove existing model if present
        if self.current_models.get("segmentation") is not None:
            idx = self.pipeline.models.index(self.current_models["segmentation"])
            self.pipeline.models.pop(idx)
            self.pipeline.model_names.pop(idx)
            self.current_models["segmentation"] = None
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.segmentation.device", "auto")
        
        # Create model service if not available
        if not hasattr(self, 'model_service'):
            from utils.model_service import ModelService
            self.model_service = ModelService(self.config)
        
        # Get the model from the service
        kwargs = {
            "device": device,
            "num_classes": num_classes
        }
        
        if model_path:
            kwargs["model_path"] = model_path
            
        model = self.model_service.get_model(model_id, **kwargs)
        
        if model is None:
            logger.error(f"Failed to create segmentation model: {model_id}")
            return False
        
        # Wrap model in adapter for safety
        from ai.model_adapter import ModelAdapter
        model_adapter = ModelAdapter(model, f"segmentation_{model_id}")
        
        self.pipeline.add_model(model_adapter, f"segmentation_{model_id}")
        self.current_models["segmentation"] = model_adapter
        return True
    
    
    
    
    
    

    
    
    
    def _set_enhancement_model(self, model_id, model_path=None, device=None, task_type='enhancement'):
        """
        Set an enhancement model (ViT-MAE, ResNet, SwinViT) for the pipeline.
        
        Args:
            model_id: ID of the enhancement model to use
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            task_type: The type of task ('enhancement', 'classification', 'segmentation', etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Setting enhancement model: {model_id} for {task_type}")
        
        # Determine which task slot to use
        task_name = "enhancement"
        if task_type == 'denoising':
            task_name = "denoising"
        elif task_type == 'reconstruction' or task_type == 'super_resolution':
            task_name = "super_resolution"
        
        # Remove existing model if present for the specified task
        if self.current_models[task_name] is not None:
            idx = self.pipeline.models.index(self.current_models[task_name])
            self.pipeline.models.pop(idx)
            self.pipeline.model_names.pop(idx)
            self.current_models[task_name] = None
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.enhancement.device", "auto")
        
        # Create model service if not available
        if not hasattr(self, 'model_service'):
            from utils.model_service import ModelService
            self.model_service = ModelService(self.config)
        
        # Get the model from the service with appropriate parameters
        kwargs = {
            "device": device,
            "task_type": task_type
        }
        
        # Add model-specific parameters
        if model_id == "novel_vit_mae_cxr":
            kwargs["encoder_only"] = task_type != 'reconstruction'
        
        if model_path:
            kwargs["model_path"] = model_path
            
        model = self.model_service.get_model(model_id, **kwargs)
        
        if model is None:
            logger.error(f"Failed to create enhancement model: {model_id}")
            return False
        
        # Wrap model in adapter for safety
        from ai.model_adapter import ModelAdapter
        model_adapter = ModelAdapter(model, f"{task_name}_{model_id}")
        
        # Add to pipeline
        self.pipeline.add_model(model_adapter, f"{task_name}_{model_id}")
        self.current_models[task_name] = model_adapter
        
        logger.info(f"Enhancement model {model_id} added to pipeline for {task_name}")
        return True
    
    def set_resnet50_rad_model(self, model_type, model_path=None, device=None, task_type='enhancement'):
        """
        Set the ResNet-50 RadImageNet model for the pipeline.
        
        Args:
            model_type: Type of ResNet model to use (typically 'novel_resnet50_rad')
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            task_type: Type of task for the model ('enhancement', 'classification', 'segmentation')
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self._set_enhancement_model(model_type, model_path, device, task_type)
    
    def set_resnet50_medical_model(self, model_type, model_path=None, device=None, task_type='enhancement'):
        """
        Set the ResNet-50 medical model for the pipeline.
        
        Args:
            model_type: Type of ResNet model to use (typically 'novel_resnet50_medical')
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            task_type: Type of task for the model ('enhancement', 'segmentation', 'classification')
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self._set_enhancement_model(model_type, model_path, device, task_type)
    
    def set_vit_mae_cxr_model(self, model_type, model_path=None, device=None, task_type='enhancement', encoder_only=True):
        """
        Set the Vision Transformer MAE model for chest X-rays in the pipeline.
        
        Args:
            model_type: Type of ViT model to use (typically 'novel_vit_mae_cxr')
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            task_type: Type of task for the model ('enhancement', 'reconstruction', 'classification')
            encoder_only: Whether to use only the encoder part (True) or full model with decoder (False)
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self._set_enhancement_model(model_type, model_path, device, task_type)
    
    def set_swinvit_model(self, model_type, model_path=None, device=None, task_type='enhancement'):
        """
        Set the SwinViT model for the pipeline.
        
        Args:
            model_type: Type of SwinViT model to use (typically 'novel_swinvit')
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            task_type: Type of task for the model ('reconstruction', 'segmentation', 'enhancement')
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self._set_enhancement_model(model_type, model_path, device, task_type)
    
    def set_denoising_model(self, model_id, model_path=None, device=None):
        """
        Set the denoising model for the pipeline.
        
        Args:
            model_id: ID of the denoising model to use
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Setting denoising model: {model_id}")
        
        # Remove existing model if present
        if self.current_models["denoising"] is not None:
            idx = self.pipeline.models.index(self.current_models["denoising"])
            self.pipeline.models.pop(idx)
            self.pipeline.model_names.pop(idx)
            self.current_models["denoising"] = None
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.denoising.device", "auto")
        
        # Create model service if not available
        if not hasattr(self, 'model_service'):
            from utils.model_service import ModelService
            self.model_service = ModelService(self.config)
        
        # Get the model from the service
        kwargs = {"device": device}
        if model_path:
            kwargs["model_path"] = model_path
            
        model = self.model_service.get_model(model_id, **kwargs)
        
        if model is None:
            logger.error(f"Failed to create denoising model: {model_id}")
            return False
        
        # Wrap model in adapter for safety
        from ai.model_adapter import ModelAdapter
        model_adapter = ModelAdapter(model, f"denoising_{model_id}")
        
        self.pipeline.add_model(model_adapter, f"denoising_{model_id}")
        self.current_models["denoising"] = model_adapter
        return True
    
    def set_super_resolution_model(self, model_id, model_path=None, device=None, scale_factor=None):
        """
        Set the super-resolution model for the pipeline.
        
        Args:
            model_id: ID of the super-resolution model to use
            model_path: Path to model weights (None to use default)
            device: Device to run inference on (None to use default)
            scale_factor: Upscaling factor (None to use default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Setting super-resolution model: {model_id}")
        
        # Remove existing model if present
        if self.current_models["super_resolution"] is not None:
            idx = self.pipeline.models.index(self.current_models["super_resolution"])
            self.pipeline.models.pop(idx)
            self.pipeline.model_names.pop(idx)
            self.current_models["super_resolution"] = None
        
        # Get scale factor from config if not provided
        if scale_factor is None:
            scale_factor = self.config.get("models.super_resolution.scale_factor", 2)
        
        # Get device from config if not provided
        if device is None:
            device = self.config.get("models.super_resolution.device", "auto")
        
        # Create model service if not available
        if not hasattr(self, 'model_service'):
            from utils.model_service import ModelService
            self.model_service = ModelService(self.config)
        
        # Get the model from the service with scale factor parameter
        kwargs = {"device": device, "scale_factor": scale_factor}
        if model_path:
            kwargs["model_path"] = model_path
            
        model = self.model_service.get_model(model_id, **kwargs)
        
        if model is None:
            logger.error(f"Failed to create super-resolution model: {model_id}")
            return False
        
        # Set scale factor on model instance if it has that attribute
        if hasattr(model, "scale_factor"):
            model.scale_factor = scale_factor
            logger.info(f"Set super-resolution scale factor to {scale_factor}")
        
        # Wrap model in adapter for safety
        from ai.model_adapter import ModelAdapter
        model_adapter = ModelAdapter(model, f"super_resolution_{model_id}_x{scale_factor}")
        
        # Add to pipeline
        self.pipeline.add_model(model_adapter, f"super_resolution_{model_id}_x{scale_factor}")
        self.current_models["super_resolution"] = model_adapter
        return True
    
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
        
        # Update config
        self.config.set("models.use_novel", self.use_novel_models)
        self.config.save()
        
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
            "artifact_removal": None,
            "enhancement": None
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
        
        # Get active models for logging
        active_models = [name for name, model in self.current_models.items() if model is not None]
        if active_models:
            logger.info(f"Processing image with active models: {', '.join(active_models)}")
        
        # Try to use available models but handle failures gracefully
        try:
            # Clone the input image to avoid modifying the original
            current_image = image.copy()
            
            # Process with each model in sequence
            for model_type, model in self.current_models.items():
                if model is not None:
                    try:
                        current_image = model.process(current_image)
                        logger.debug(f"Processed with {model_type} successfully")
                    except Exception as e:
                        logger.error(f"Error processing with {model_type}: {e}")
                        # Continue with the current image if a model fails
            
            # Make sure the result is valid
            if current_image is None or not isinstance(current_image, np.ndarray):
                logger.error("Pipeline produced invalid output, falling back to original image")
                return image
            
            # Make sure result shape matches input, resizing if necessary
            if current_image.shape[:2] != image.shape[:2]:
                logger.warning(f"Output shape {current_image.shape[:2]} doesn't match input {image.shape[:2]}, resizing")
                from data.processing.transforms import resize_image
                current_image = resize_image(current_image, (image.shape[1], image.shape[0]))
            
            return current_image
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