"""
Minimal model downloader stub for the medical image enhancement application.
This is a placeholder that doesn't actually download models but provides the
expected interface for other parts of the application.
"""
import os
import logging
from pathlib import Path
import json

from utils.config import Config

logger = logging.getLogger(__name__)

class ModelDownloader:
    """
    Stub for model downloading functionality.
    This version only looks for local models and doesn't download anything.
    """
    
    def __init__(self, base_dir=None, config=None):
        """
        Initialize the model downloader stub.
        
        Args:
            base_dir: Base directory for model weights storage
            config: Configuration object
        """
        self.config = config or Config()
        
        # Set up model directories
        if base_dir is None:
            self.base_dir = Path(self.config.get("paths.model_weights_dir", "weights"))
        else:
            self.base_dir = Path(base_dir)
            
        # Create category directories
        self.foundational_dir = self.base_dir / "foundational"
        self.novel_dir = self.base_dir / "novel"
        
        # Model registry - just contains the basic info about available models
        self.MODEL_REGISTRY = {
            "dncnn_denoiser": {
                "file_name": "denoising/dncnn_25.pth",
                "description": "DnCNN model for grayscale blind denoising",
                "category": "foundational"
            },
            "edsr_super_resolution": {
                "file_name": "super_resolution/RealESRGAN_x2.pth",
                "description": "RealESRGAN model for super-resolution",
                "category": "foundational"
            },
            "unet_artifact_removal": {
                "file_name": "artifact_removal/G_ema_ep_82.pth",
                "description": "U-Net GAN model for artifact removal",
                "category": "foundational"
            }
        }
    
    def get_model_path(self, model_id):
        """
        Get the local path for a model by ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Path: Path to the model weights file or None if not found
        """
        if model_id not in self.MODEL_REGISTRY:
            logger.error(f"Model ID not found in registry: {model_id}")
            return None
        
        model_info = self.MODEL_REGISTRY[model_id]
        category = model_info.get("category", "foundational")
        file_name = model_info.get("file_name", f"{model_id}.pth")
        
        # Determine the directory based on category
        if category == "novel":
            model_dir = self.novel_dir
        else:
            model_dir = self.foundational_dir
            
        model_path = model_dir / file_name
        
        # Check if the file exists
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None
            
        return model_path
    
    def download_model(self, model_id, force=False):
        """
        This method doesn't actually download anything, just returns the local path.
        
        Args:
            model_id: ID of the model to "download"
            force: Ignored parameter
            
        Returns:
            Path: Path to the local model file or None if not found
        """
        logger.info(f"Model downloading disabled - using local model: {model_id}")
        return self.get_model_path(model_id)
    
    def download_all_models(self, category=None, force=False):
        """
        This method doesn't actually download anything, just returns success status.
        
        Args:
            category: Filter by category ("foundational", "novel", or None for all)
            force: Ignored parameter
            
        Returns:
            dict: Dictionary of model_id -> True if local file exists
        """
        logger.info("Model downloading disabled - using only local models")
        results = {}
        
        for model_id in self.MODEL_REGISTRY:
            # Skip if category doesn't match
            if category and self.MODEL_REGISTRY[model_id].get("category") != category:
                continue
                
            # Check if the file exists locally
            path = self.get_model_path(model_id)
            results[model_id] = path is not None
        
        return results
    
    def list_available_models(self, category=None, downloaded_only=False):
        """
        List available models in the registry.
        
        Args:
            category: Filter by category ("foundational", "novel", or None for all)
            downloaded_only: Whether to list only downloaded models
            
        Returns:
            list: List of model information dictionaries
        """
        result = []
        
        for model_id, model_info in self.MODEL_REGISTRY.items():
            # Skip if category doesn't match
            if category and model_info.get("category") != category:
                continue
                
            # Check if the model is downloaded
            model_path = self.get_model_path(model_id)
            is_downloaded = model_path is not None and model_path.exists()
            
            # Skip if we only want downloaded models and this one isn't
            if downloaded_only and not is_downloaded:
                continue
                
            # Add to result
            result.append({
                "id": model_id,
                "description": model_info.get("description", ""),
                "category": model_info.get("category", "foundational"),
                "downloaded": is_downloaded,
                "file_name": model_info.get("file_name", f"{model_id}.pth"),
                "path": str(model_path) if model_path else None
            })
        
        return result