"""
Model adapter for the medical image enhancement application.
Provides a reliable wrapper around AI models to ensure they always return valid results.
"""
import os
import logging
import numpy as np
import traceback

logger = logging.getLogger(__name__)

class ModelAdapter:
    """
    Adapter for AI models that ensures they always return valid results.
    Implements error handling, fallbacks, and validation to protect the application
    from model failures.
    """
    
    def __init__(self, model, name="unknown"):
        """
        Initialize the model adapter.
        
        Args:
            model: The AI model to wrap
            name: Name of the model for logging
        """
        self.model = model
        self.name = name
        self.enabled = True
    
    
    def process(self, image):
        """
        Process an image with robust error handling.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            numpy.ndarray: Processed image or original if processing fails
        """
        if not self.enabled or self.model is None:
            logger.debug(f"Model {self.name} is disabled or not initialized, skipping")
            return image
        
        try:
            # Create a copy to avoid modifying the original
            image_copy = image.copy()
            
            # Process the image
            logger.debug(f"Processing with model {self.name}")
            result = self.model.process(image_copy)
            
            # Validate the result
            if result is None:
                logger.error(f"Model {self.name} returned None, using original image")
                return image
                
            if not isinstance(result, np.ndarray):
                logger.error(f"Model {self.name} returned non-numpy result: {type(result)}, using original image")
                return image
                
            if not np.isfinite(result).all():
                logger.error(f"Model {self.name} returned non-finite values, using original image")
                return image
                
            # Check dimensions and fix if needed
            if result.ndim != image.ndim:
                logger.warning(f"Model {self.name} returned result with different dimensions: expected {image.ndim}, got {result.ndim}")
                
                # Convert 2D grayscale to 3D if needed
                if result.ndim == 2 and image.ndim == 3:
                    result = np.stack([result] * image.shape[2], axis=2)
                # Convert 3D to 2D if needed
                elif result.ndim == 3 and image.ndim == 2:
                    result = np.mean(result, axis=2)
            
            # Success
            logger.debug(f"Model {self.name} processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing with model {self.name}: {e}")
            logger.debug(traceback.format_exc())
            return image
    
    
    def enable(self):
        """Enable the model."""
        self.enabled = True
    
    def disable(self):
        """Disable the model."""
        self.enabled = False
    
    def __call__(self, image):
        """Make the adapter callable like the underlying model."""
        return self.process(image)