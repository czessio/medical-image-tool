"""
Base model class for the medical image enhancement application.
Defines the interface for all AI models used in the application.
"""
import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all AI models in the application."""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the base model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.model = None
        self.initialized = False
        
        # Initialize if model_path is provided
        if model_path:
            self.initialize()
    
    def _determine_device(self, device):
        """
        Determine the device to use for inference.
        
        Args:
            device: Requested device ('cpu', 'cuda', or None for auto-detection)
            
        Returns:
            str: 'cuda' if available and requested, otherwise 'cpu'
        """
        if device is None or device == 'auto':
            # Auto-detect the best available device
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        
        # Use the requested device if possible
        if device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            return 'cuda'
        
        # Fall back to CPU
        return 'cpu'
    
    
    
    def initialize(self):
        """
        Initialize the model by loading weights and preparing for inference.
        """
        if self.initialized:
            logger.debug("Model already initialized")
            return
    
        # For testing: Allow initialization without a model path for test models
        if self.model_path is None:
            try:
                self._load_model()
                self.initialized = True
                logger.info("Model initialized with no weights (test mode)")
                return
            except Exception as e:
                logger.error(f"Error initializing model in test mode: {e}")
                raise
    
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
        try:
            self._load_model()
            self.initialized = True
            logger.info(f"Model initialized from {self.model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    
    
    @abstractmethod
    def _load_model(self):
        """
        Load the model from the specified path.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def preprocess(self, image):
        """
        Preprocess the input image before inference.
        Must be implemented by subclasses.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image in the format expected by the model
        """
        pass
    
    @abstractmethod
    def inference(self, preprocessed_image):
        """
        Run inference on the preprocessed image.
        Must be implemented by subclasses.
        
        Args:
            preprocessed_image: Preprocessed input data
            
        Returns:
            Raw model output
        """
        pass
    
    @abstractmethod
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output.
        Must be implemented by subclasses.
        
        Args:
            model_output: Raw output from the model
            original_image: Original input image (if needed for reference)
            
        Returns:
            Processed output image as numpy array
        """
        pass
    
    def process(self, image):
        """
        Process an image through the full pipeline: preprocess -> inference -> postprocess.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed output image as numpy array
        """
        if not self.initialized:
            self.initialize()
        
        preprocessed = self.preprocess(image)
        output = self.inference(preprocessed)
        result = self.postprocess(output, image)
        
        return result
    
    def __call__(self, image):
        """
        Make the model callable directly.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed output image as numpy array
        """
        return self.process(image)