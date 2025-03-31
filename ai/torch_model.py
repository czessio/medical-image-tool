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
        Load the PyTorch model from the specified path.
        """
        # Create PyTorch device
        self.torch_device = torch.device(self.device)
        
        # Create model architecture (to be implemented by subclasses)
        self.model = self._create_model_architecture()
        
        # Load weights if a path is provided
        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location=self.torch_device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                logger.error(f"Error loading model weights: {e}")
                raise
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.torch_device)
        
        logger.info(f"PyTorch model loaded on {self.device}")
    
    
    
    
    
    
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
            
        # Move to device
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
        with torch.no_grad():
            output = self.model(preprocessed_tensor)
        return output
    
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