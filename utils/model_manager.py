"""
Model manager for medical image enhancement application.
Handles loading, caching, and inference with AI models.
"""
import os
import logging
import time
from pathlib import Path
import numpy as np
import threading
from queue import Queue
import weakref

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.model_registry import ModelRegistry
from utils.config import Config
from utils.model_downloader import ModelDownloader

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Cache for loaded models to avoid reloading the same model.
    Uses weak references to allow models to be garbage collected when not in use.
    """
    
    def __init__(self, max_size=5):
        """
        Initialize the model cache.
        
        Args:
            max_size: Maximum number of models to keep in cache
        """
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get(self, key):
        """
        Get a model from the cache.
        
        Args:
            key: Cache key (typically model_id)
            
        Returns:
            Model instance or None if not in cache
        """
        with self.lock:
            # Check if key exists in cache
            if key not in self.cache:
                return None
                
            # Get the weak reference
            ref = self.cache[key]
            model = ref()
            
            # If the model has been garbage collected, remove it from cache
            if model is None:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return None
                
            # Update access time
            self.access_times[key] = time.time()
            
            return model
    
    def put(self, key, model):
        """
        Add a model to the cache.
        
        Args:
            key: Cache key (typically model_id)
            model: Model instance
        """
        with self.lock:
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.max_size:
                self._remove_lru()
                
            # Add model to cache using weak reference
            self.cache[key] = weakref.ref(model)
            self.access_times[key] = time.time()
    
    def _remove_lru(self):
        """Remove the least recently used item from the cache."""
        if not self.access_times:
            return
            
        # Find the least recently used key
        lru_key = min(self.access_times, key=self.access_times.get)
        
        # Remove it from cache
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


class ModelManager:
    """
    Manager for AI models used in the application.
    
    Features:
    - Model loading with automatic downloading
    - Model caching to avoid reloading
    - Batch inference for efficiency
    - Automatic fallback to CPU if GPU is not available
    """
    
    def __init__(self, config=None):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.downloader = ModelDownloader(
            base_dir=self.config.get("paths.model_weights_dir"),
            config=self.config
        )
        self.cache = ModelCache(max_size=5)
        
        # Check PyTorch availability
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch is not available. Models will not work.")
    
    def get_model(self, model_id, auto_download=None):
        """
        Get a model by ID, loading it if necessary.
        
        Args:
            model_id: ID of the model to get
            auto_download: Whether to download the model if not found (None to use config setting)
            
        Returns:
            Model instance or None if not available
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is not available. Cannot load model.")
            return None
            
        # Check cache first
        model = self.cache.get(model_id)
        if model is not None:
            logger.debug(f"Model {model_id} found in cache")
            return model
            
        # If not in cache, try to load it
        logger.info(f"Loading model: {model_id}")
        
        # Get the model path
        model_path = self.downloader.get_model_path(model_id)
        
        
        
        
        if model_path is None or not model_path.exists():
            logger.error(f"Model file not found: {model_id}")
            return None
        
        
        
        
        
        
        # Create the model
        model = ModelRegistry.create(model_id, model_path=str(model_path))
        
        if model is None:
            logger.error(f"Failed to create model: {model_id}")
            return None
        
        # Add to cache
        self.cache.put(model_id, model)
        
        return model
    
    def process_image(self, image, model_id, **kwargs):
        """
        Process an image with a model.
        
        Args:
            image: Input image as numpy array
            model_id: ID of the model to use
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Processed image as numpy array or None if processing failed
        """
        # Get the model
        model = self.get_model(model_id)
        
        if model is None:
            logger.error(f"Model not available: {model_id}")
            return None
        
        # Process the image
        try:
            result = model.process(image, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error processing image with model {model_id}: {e}")
            return None
    
    def get_available_models(self, category=None, downloaded_only=False):
        """
        Get a list of available models.
        
        Args:
            category: Filter by category ("foundational", "novel", or None for all)
            downloaded_only: Whether to include only downloaded models
            
        Returns:
            list: List of model information dictionaries
        """
        return self.downloader.list_available_models(category, downloaded_only)
    
    def clear_cache(self):
        """Clear the model cache."""
        self.cache.clear()
        
    def download_all_models(self, category=None, force=False):
        """
        Download all models.
        
        Args:
            category: Filter by category ("foundational", "novel", or None for all)
            force: Whether to force redownload even if the model is already downloaded
            
        Returns:
            dict: Dictionary of model_id -> download success (True/False)
        """
        return self.downloader.download_all_models(category, force)