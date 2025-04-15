"""
Model service for the medical image enhancement application.
Provides a unified interface for model management, including loading,
caching, initialization, and model availability checking.
"""
import os
import logging
import time
from pathlib import Path
import threading
from queue import Queue
import weakref
import json
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.model_registry import ModelRegistry
from utils.config import Config

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


class ModelService:
    """
    Unified service for model management in the application.
    
    This service centralizes all model-related functionality:
    - Model discovery and registration
    - Model initialization and loading
    - Model caching and lifecycle management
    - Model availability checking
    - Configuration management for models
    
    It replaces the separate ModelDownloader, ModelInitializer, and ModelManager
    classes with a more cohesive design.
    """
    
    # Dictionary defining model categories and types
    MODEL_REGISTRY = {
        "denoising": {
            "foundational": {
                "dncnn_denoiser": {
                    "description": "DnCNN model for grayscale blind denoising",
                    "default_path": "weights/foundational/denoising/dncnn_25.pth",
                    "alt_paths": [
                        "weights/foundational/denoising/dncnn_gray_blind.pth"
                    ]
                }
            },
            "novel": {
                "novel_diffusion_denoiser": {
                    "description": "Diffusion-based denoising model",
                    "default_path": "weights/novel/denoising/diffusion_denoiser.pth"
                }
            }
        },
        "super_resolution": {
            "foundational": {
                "edsr_super_resolution": {
                    "description": "Enhanced Deep Super-Resolution model",
                    "default_path": "weights/foundational/super_resolution/RealESRGAN_x2.pth",
                    "alt_paths": [
                        "weights/foundational/super_resolution/edsr_x2.pt",
                        "weights/foundational/super_resolution/RealESRGAN_x4.pth",
                        "weights/foundational/super_resolution/RealESRGAN_x8.pth"
                    ]
                }
            },
            "novel": {
                "novel_restormer": {
                    "description": "Restormer model for super-resolution",
                    "default_path": "weights/novel/super_resolution/restormer_sr.pth"
                },
                "novel_swinir_super_resolution": {
                    "description": "SwinIR model for super-resolution",
                    "default_path": "weights/novel/super_resolution/swinir_sr.pth"
                }
            }
        },
        "artifact_removal": {
            "foundational": {
                "unet_artifact_removal": {
                    "description": "U-Net model for artifact removal",
                    "default_path": "weights/foundational/artifact_removal/G_ema_ep_82.pth"
                }
            },
            "novel": {
                "novel_stylegan_artifact_removal": {
                    "description": "StyleGAN-based artifact removal",
                    "default_path": "weights/novel/artifact_removal/stylegan_artifact_removal.pth"
                }
            }
        },
        "enhancement": {
            "novel": {
                "novel_swinvit": {
                    "description": "Swin Vision Transformer for image enhancement",
                    "default_path": "weights/novel/enhancement/model_swinvit.pt"
                },
                "novel_resnet50_medical": {
                    "description": "ResNet-50 trained on 23 medical datasets",
                    "default_path": "weights/novel/enhancement/resnet_50_23dataset.pt"
                },
                "novel_resnet50_rad": {
                    "description": "ResNet-50 trained on RadImageNet",
                    "default_path": "weights/novel/enhancement/ResNet50.pt"
                },
                "novel_vit_mae_cxr": {
                    "description": "ViT-MAE trained on chest X-rays",
                    "default_path": "weights/novel/enhancement/vit-b_CXR_0.5M_mae.pth"
                }
            }
        }
    }
    
    def __init__(self, config=None):
        """
        Initialize the model service.
        
        Args:
            config: Configuration object or None to use default
        """
        self.config = config or Config()
        self.cache = ModelCache(max_size=8)  # Increased cache size for more models
        
        # Initialize model directories
        self.base_dir = Path(self.config.get("paths.model_weights_dir", "weights"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize category directories
        for category in self.MODEL_REGISTRY.keys():
            category_dir = self.base_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize type directories within each category
            if "foundational" in self.MODEL_REGISTRY[category]:
                (self.base_dir / "foundational" / category).mkdir(parents=True, exist_ok=True)
            if "novel" in self.MODEL_REGISTRY[category]:
                (self.base_dir / "novel" / category).mkdir(parents=True, exist_ok=True)
        
        # Check PyTorch availability
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch is not available. Models will not work.")
    
    def initialize(self):
        """
        Initialize the model service at application startup.
        Checks for model availability and prepares the model registry.
        
        Returns:
            dict: Status of model initialization with categories as keys
        """
        logger.info("Initializing model service")
        
        status = {
            "foundational": self.check_category_availability("foundational"),
            "novel": self.check_category_availability("novel")
        }
        
        # Log initialization status
        for category, models in status.items():
            available_count = sum(1 for m in models.values() if m)
            total_count = len(models)
            logger.info(f"{category.capitalize()} models: {available_count}/{total_count} available")
        
        return status
    
    def check_category_availability(self, category="foundational"):
        """
        Check which models are available in a category.
        
        Args:
            category: Model category ("foundational" or "novel")
            
        Returns:
            dict: Model IDs mapped to availability status (True/False)
        """
        result = {}
        
        for model_type, categories in self.MODEL_REGISTRY.items():
            if category in categories:
                for model_id in categories[category].keys():
                    model_path = self.resolve_model_path(model_id)
                    result[model_id] = model_path is not None and Path(model_path).exists()
        
        return result
    
    def check_model_availability(self, model_id):
        """
        Check if a specific model is available.
        
        Args:
            model_id: Model identifier
            
        Returns:
            bool: True if the model is available, False otherwise
        """
        model_path = self.resolve_model_path(model_id)
        return model_path is not None and Path(model_path).exists()
    
    def resolve_model_path(self, model_id):
        """
        Resolve the path to a model's weights file.
        
        This method handles the logic of finding model files by:
        1. Checking the config for a specified path
        2. Checking the default path in MODEL_REGISTRY
        3. Checking alternative paths in MODEL_REGISTRY
        
        Args:
            model_id: Model identifier
            
        Returns:
            str: Path to the model weights file or None if not found
        """
        # First check if the model ID is in the registry
        for model_type, categories in self.MODEL_REGISTRY.items():
            for category in categories.keys():
                if model_id in categories[category]:
                    # Check if there's a path in the config
                    config_path = self.config.get(f"models.{model_type}.{category}.{model_id}.model_path")
                    if config_path and Path(config_path).exists():
                        return config_path
                    
                    # Try the default path from the registry
                    model_info = categories[category][model_id]
                    default_path = model_info.get("default_path")
                    if default_path and Path(default_path).exists():
                        return default_path
                    
                    # Try alternative paths if available
                    alt_paths = model_info.get("alt_paths", [])
                    for alt_path in alt_paths:
                        if Path(alt_path).exists():
                            # Update config with the found path
                            self.config.set(f"models.{model_type}.{category}.{model_id}.model_path", alt_path)
                            self.config.save()
                            return alt_path
                    
                    # Log that we couldn't find the model
                    logger.warning(f"Could not find weights for model {model_id}")
                    return None
        
        # If we get here, the model ID wasn't found in the registry
        logger.error(f"Unknown model ID: {model_id}")
        return None
    
    def get_model(self, model_id, **kwargs):
        """
        Get a model instance by ID, loading it if necessary.
        
        Args:
            model_id: Model identifier
            **kwargs: Additional parameters to pass to the model constructor
            
        Returns:
            Model instance or None if the model cannot be loaded
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is not available. Cannot load model.")
            return None
        
        # Check cache first
        model = self.cache.get(model_id)
        if model is not None:
            logger.debug(f"Model {model_id} found in cache")
            return model
        
        # Resolve the model path
        model_path = self.resolve_model_path(model_id)
        if model_path is None:
            logger.error(f"Could not resolve path for model {model_id}")
            return None
        
        # Create the model
        logger.info(f"Loading model {model_id} from {model_path}")
        try:
            model = ModelRegistry.create(model_id, model_path=model_path, **kwargs)
            
            if model is None:
                logger.error(f"Failed to create model {model_id}")
                return None
            
            # Add to cache
            self.cache.put(model_id, model)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def process_image(self, image, model_id, **kwargs):
        """
        Process an image using the specified model.
        
        Args:
            image: Input image as numpy array
            model_id: Model identifier
            **kwargs: Additional parameters to pass to the model's process method
            
        Returns:
            numpy.ndarray: Processed image or None if processing failed
        """
        # Get the model
        model = self.get_model(model_id)
        
        if model is None:
            logger.error(f"Model {model_id} not available")
            return None
        
        # Process the image
        try:
            result = model.process(image, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error processing image with model {model_id}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the model cache."""
        self.cache.clear()
        logger.info("Model cache cleared")
    
    def get_available_models(self, category=None, model_type=None):
        """
        Get a list of available models, optionally filtered by category and type.
        
        Args:
            category: Filter by category ("foundational", "novel", or None for all)
            model_type: Filter by model type ("denoising", "super_resolution", etc.)
            
        Returns:
            list: List of model information dictionaries
        """
        result = []
        
        for curr_type, categories in self.MODEL_REGISTRY.items():
            # Skip if type doesn't match
            if model_type is not None and curr_type != model_type:
                continue
                
            for curr_category in categories.keys():
                # Skip if category doesn't match
                if category is not None and curr_category != category:
                    continue
                    
                for model_id, model_info in categories[curr_category].items():
                    # Check if the model is available
                    model_path = self.resolve_model_path(model_id)
                    is_available = model_path is not None and Path(model_path).exists()
                    
                    # Add to result
                    result.append({
                        "id": model_id,
                        "description": model_info.get("description", ""),
                        "category": curr_category,
                        "type": curr_type,
                        "available": is_available,
                        "path": model_path
                    })
        
        return result
    
    def get_model_info(self, model_id):
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            dict: Model information or None if model not found
        """
        # Search for the model in the registry
        for model_type, categories in self.MODEL_REGISTRY.items():
            for category in categories.keys():
                if model_id in categories[category]:
                    model_info = categories[category][model_id].copy()
                    model_info["id"] = model_id
                    model_info["category"] = category
                    model_info["type"] = model_type
                    
                    # Check availability
                    model_path = self.resolve_model_path(model_id)
                    model_info["available"] = model_path is not None and Path(model_path).exists()
                    model_info["path"] = model_path
                    
                    return model_info
        
        return None