"""
Inference pipeline for the medical image enhancement application.
Manages the execution of multiple AI models in sequence.
"""
import logging
import numpy as np
from pathlib import Path

from .model_registry import ModelRegistry
from utils.memory_monitor import memory_monitor  # Import the memory monitor

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
    
    def process_batch(self, images, batch_size=4):
        """
        Process a batch of images through the pipeline with specified batch size.
        
        Args:
            images: List of input images as numpy arrays
            batch_size: Maximum number of images to process at once
            
        Returns:
            list: List of processed output images
        """
        if not self.models:
            logger.warning("No models in pipeline, returning original images")
            return images.copy()
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}, size: {len(batch)}")
            
            # Process the batch through each model in sequence
            current_batch = [img.copy() for img in batch]
            
            for model_idx, (model, name) in enumerate(zip(self.models, self.model_names)):
                logger.debug(f"Running {name} ({model_idx+1}/{len(self.models)}) on batch")
                try:
                    # Start memory monitoring for this model
                    memory_monitor.start_monitoring()
                    
                    # If the model supports batched processing, use it
                    if hasattr(model, 'process_batch'):
                        current_batch = model.process_batch(current_batch)
                    else:
                        # Otherwise, process images individually
                        current_batch = [model.process(img) for img in current_batch]
                    
                    # Stop memory monitoring
                    memory_monitor.stop_monitoring()
                    
                except Exception as e:
                    logger.error(f"Error in model {name} during batch processing: {e}")
                    # Continue with the current batch if a model fails
                    
                    # Stop memory monitoring on error
                    memory_monitor.stop_monitoring()
            
            # Add processed batch to results
            results.extend(current_batch)
            
            # Collect garbage after processing a batch
            memory_monitor.collect_garbage()
        
        return results
    
    def clear(self):
        """Clear all models from the pipeline."""
        self.models = []
        self.model_names = []
        logger.debug("Pipeline cleared")
    
    def estimate_optimal_batch_size(self, sample_image):
        """
        Estimate the optimal batch size based on available memory and sample image.
        
        Args:
            sample_image: A sample image to use for estimation
            
        Returns:
            int: Estimated optimal batch size
        """
        # Use memory monitor to estimate batch size
        batch_size = memory_monitor.estimate_batch_size(sample_image, target_memory_usage=0.7)
        
        # Ensure batch size is at least 1
        batch_size = max(1, batch_size)
        
        logger.info(f"Estimated optimal batch size: {batch_size}")
        return batch_size
    
    def __call__(self, image):
        """
        Make the pipeline callable directly.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            numpy.ndarray: Processed output image
        """
        return self.process(image)