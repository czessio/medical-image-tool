"""
Model initializer for the medical image enhancement application.
Sets up required model weights during application startup.
"""
import os
import logging
from pathlib import Path

from utils.config import Config
from utils.model_downloader import ModelDownloader

logger = logging.getLogger(__name__)

class ModelInitializer:
    """
    Initializes models for the application.
    
    This version only checks for local models and doesn't try to download anything.
    """
    
    def __init__(self, config=None):
        """
        Initialize the model initializer.
        
        Args:
            config: Configuration object or None to use default
        """
        self.config = config or Config()
        self.model_downloader = ModelDownloader(
            base_dir=self.config.get("paths.model_weights_dir"),
            config=self.config
        )
    
    def check_model_availability(self, model_type="foundational"):
        """
        Check if models are available locally.
        
        Args:
            model_type: Type of models to check ("foundational", "novel", or None for all)
            
        Returns:
            dict: Dictionary of model_id -> availability (True/False)
        """
        models = self.model_downloader.list_available_models(category=model_type)
        results = {}
        
        for model in models:
            model_id = model["id"]
            model_path = self.model_downloader.get_model_path(model_id)
            results[model_id] = model_path is not None and model_path.exists()
        
        return results
    
    def initialize_for_application(self, download_missing=False):
        """
        Initialize models for application startup.
        
        Args:
            download_missing: Ignored parameter, no downloading will happen
            
        Returns:
            dict: Dictionary of model availability status
        """
        logger.info("Initializing models for application startup")
        
        # Ensure weights directories exist
        weights_dir = Path(self.config.get("paths.model_weights_dir", "weights"))
        foundational_dir = weights_dir / "foundational"
        novel_dir = weights_dir / "novel"
        
        # Create category directories if they don't exist
        foundational_dir.mkdir(parents=True, exist_ok=True)
        (foundational_dir / "denoising").mkdir(exist_ok=True)
        (foundational_dir / "super_resolution").mkdir(exist_ok=True)
        (foundational_dir / "artifact_removal").mkdir(exist_ok=True)
        novel_dir.mkdir(parents=True, exist_ok=True)
        
        # Check which models are available
        foundational_status = self.check_model_availability("foundational")
        novel_status = self.check_model_availability("novel")
        
        # Combine status
        all_status = {**foundational_status, **novel_status}
        
        return all_status