"""
PyTorch model implementation for the medical image enhancement application.
Extends the BaseModel with PyTorch-specific functionality.
"""
import os
import logging
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class TorchModel(BaseModel):
    """Base class for PyTorch models."""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the PyTorch model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        super().__init__(model_path, device)
        self.torch_device = None
    
    
    
    
    
    
    def _load_model(self):
        """
        Load the PyTorch model from the specified path with improved error handling.
        """
        # Create PyTorch device
        self.torch_device = torch.device(self.device)
        
        # Create model architecture (to be implemented by subclasses)
        self.model = self._create_model_architecture()
        
        # Load weights if a path is provided
        if self.model_path:
            try:
                logger.info(f"Loading model weights from: {self.model_path}")
                state_dict = torch.load(self.model_path, map_location=self.torch_device)
                
                # Handle different formats of saved models
                if isinstance(state_dict, dict):
                    # Some models save the state dict under specific keys
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'params_ema' in state_dict:
                        state_dict = state_dict['params_ema']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                    
                    # Don't use the weights if they're not a state dict at this point
                    if not any(isinstance(v, torch.Tensor) for v in state_dict.values()):
                        logger.warning("Invalid state dict format, using model with random weights")
                        return
                
                # Try to load state dict with strict=False to ignore missing keys
                try:
                    # Use strict=False to ignore missing or unexpected keys
                    result = self.model.load_state_dict(state_dict, strict=False)
                    
                    # Log missing and unexpected keys for debugging
                    if result.missing_keys:
                        logger.warning(f"Missing keys in state_dict: {result.missing_keys}")
                    if result.unexpected_keys:
                        logger.warning(f"Unexpected keys in state_dict: {result.unexpected_keys}")
                    
                    logger.info(f"Model weights loaded successfully with non-strict matching")
                    
                except Exception as e:
                    logger.error(f"Error loading state dict with non-strict matching: {e}")
                    # If that fails, try to use a custom loading function if available
                    if hasattr(self, '_custom_load_state_dict'):
                        logger.info("Attempting custom state dict loading")
                        self._custom_load_state_dict(state_dict)
                    else:
                        raise
                        
            except Exception as e:
                logger.error(f"Error loading model weights: {e}")
                logger.warning("Using model with random weights")
    
    
    
    
    
    
    def _create_model_architecture(self):
        """
        Create the model architecture.
        Must be implemented by subclasses.
        
        Returns:
            torch.nn.Module: Model architecture
        """
        raise NotImplementedError("Subclasses must implement _create_model_architecture")
    
    def preprocess(self, image):
        """
        Preprocess the input image for PyTorch model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to float32 if not already
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Ensure values are in [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(image)
        
        # Add batch dimension if missing
        if len(tensor.shape) == 3:  # [H, W, C]
            # Move channels to first dimension
            tensor = tensor.permute(2, 0, 1)  # [C, H, W]
            tensor = tensor.unsqueeze(0)  # [1, C, H, W]
        elif len(tensor.shape) == 2:  # [H, W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
        # Move to device - use to() with the device object not the string
        # This ensures the device type and index match exactly
        tensor = tensor.to(self.torch_device)
        
        return tensor
    
    def inference(self, preprocessed_tensor):
            """
            Run inference with the PyTorch model.
            
            Args:
                preprocessed_tensor: Preprocessed input tensor
                
            Returns:
                torch.Tensor: Model output tensor
            """
            # Make sure the model and preprocessed tensor are on the same device
            current_device = next(self.model.parameters()).device
            if preprocessed_tensor.device != current_device:
                preprocessed_tensor = preprocessed_tensor.to(current_device)
                
            with torch.no_grad():
                try:
                    output = self.model(preprocessed_tensor)
                    return output
                except Exception as e:
                    logger.error(f"Error during model inference: {e}")
                    # Return input as fallback if error occurs during inference
                    logger.warning("Returning input tensor as fallback due to inference error")
                    return preprocessed_tensor
    
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output to produce the final image.
        
        Args:
            model_output: Raw output from the model as tensor
            original_image: Original input image (if needed for reference)
            
        Returns:
            numpy.ndarray: Processed output image
        """
        # Move to CPU and convert to numpy
        output = model_output.squeeze().cpu().numpy()
        
        # If output has channels, move them to last dimension
        if len(output.shape) == 3:
            output = np.transpose(output, (1, 2, 0))
        
        # Ensure output values are in [0, 1]
        output = np.clip(output, 0, 1)
        
        return output