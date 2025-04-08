# Update process method in ai/cleaning/inference/cleaning_pipeline.py

import logging
logger = logging.getLogger(__name__)


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
        result = self.pipeline.process(image)
        return result
    except Exception as e:
        logger.error(f"Error in cleaning pipeline: {e}")
        logger.warning("Falling back to original image due to pipeline error")
        return image

# Also update the toggle_model_type method to try to use available models

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

# Instructions:
# 1. Open ai/cleaning/inference/cleaning_pipeline.py
# 2. Update the process method to handle failures gracefully
# 3. Update the toggle_model_type method to revert to previous setting if no models are available