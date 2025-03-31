"""
Inference pipeline for the medical image enhancement application.
Manages the execution of multiple AI models in sequence.
"""
import logging
import numpy as np
from pathlib import Path

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class InferencePipeline:
    """
    Pipeline for running inference with multiple models.
    Allows chaining models together where the output of one model becomes the input to the next.
    """
    
    def __init__(self):
        """Initialize an empty inference pipeline."""
        self.models = []
        self.model_names = []
    
    def add_model(self, model, name=None):
        """
        Add a model to the pipeline.
        
        Args:
            model: Model instance or tuple (model_type, model_path, device)
            name: Optional name for the model
        
        Returns:
            bool: True if the model was added successfully
        """
        if isinstance(model, tuple):
            # Model specified as (model_type, model_path, device)
            model_type, model_path, device = model
            model_instance = ModelRegistry.create(model_type, model_path=model_path, device=device)
            if model_instance is None:
                logger.error(f"Failed to create model of type {model_type}")
                return False
            model = model_instance
        
        self.models.append(model)
        self.model_names.append(name or f"Model_{len(self.models)}")
        logger.info(f"Added model to pipeline: {self.model_names[-1]}")
        return True
    
    def process(self, image):
        """
        Process an image through the pipeline.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            numpy.ndarray: Processed output image
        """
        if not self.models:
            logger.warning("No models in pipeline, returning original image")
            return image
        
        current_image = image.copy()
        
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            logger.debug(f"Running {name} ({i+1}/{len(self.models)})")
            try:
                current_image = model.process(current_image)
            except Exception as e:
                logger.error(f"Error in model {name}: {e}")
                # Continue with the current image if a model fails
        
        return current_image
    
    def clear(self):
        """Clear all models from the pipeline."""
        self.models = []
        self.model_names = []
        logger.debug("Pipeline cleared")
    
    def __call__(self, image):
        """
        Make the pipeline callable directly.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            numpy.ndarray: Processed output image
        """
        return self.process(image)