"""
Model initializer for the medical image enhancement application.
Downloads and sets up all required model weights during application startup.
"""
import os
import sys
import logging
import argparse
import threading
from pathlib import Path

from utils.config import Config
from utils.model_downloader import ModelDownloader
from utils.model_manager import ModelManager

logger = logging.getLogger(__name__)

class ModelInitializer:
    """
    Initializes and downloads models for the application.
    
    Features:
    - Checks for model availability
    - Downloads missing models in background
    - Validates model checksums
    - Sets up model directories
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
        self.model_manager = ModelManager(self.config)
        
        # Keep track of download threads
        self.download_threads = {}
    
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
    
    def download_models(self, model_type=None, force=False, block=False):
        """
        Download models in the background.
        
        Args:
            model_type: Type of models to download ("foundational", "novel", or None for all)
            force: Whether to force re-download even if files exist
            block: Whether to block until download is complete
            
        Returns:
            threading.Thread: Download thread or None if already running
        """
        # Check if a download thread is already running for this model type
        if model_type in self.download_threads and self.download_threads[model_type].is_alive():
            logger.warning(f"Download already in progress for {model_type} models")
            return None
        
        # Create download thread
        thread = threading.Thread(
            target=self._download_models_thread,
            args=(model_type, force),
            daemon=True  # Make thread a daemon so it doesn't block application exit
        )
        
        # Start thread
        logger.info(f"Starting download thread for {model_type} models")
        thread.start()
        
        # Store thread
        self.download_threads[model_type] = thread
        
        # Wait for completion if requested
        if block:
            thread.join()
        
        return thread
    
    def _download_models_thread(self, model_type, force):
        """Thread function for downloading models."""
        try:
            # Download models
            results = self.model_downloader.download_all_models(category=model_type, force=force)
            
            # Log results
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Downloaded {success_count}/{len(results)} {model_type} models")
            
            # Log failed models
            failed = [model_id for model_id, success in results.items() if not success]
            if failed:
                logger.warning(f"Failed to download models: {', '.join(failed)}")
                
        except Exception as e:
            logger.error(f"Error in download thread: {e}")
    
    
    
    
    
    def initialize_for_application(self, download_missing=False):
        """
        Initialize models for application startup.
        
        Args:
            download_missing: Whether to download missing models
            
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
        
        # Download missing models if requested
        if download_missing:
            missing_foundational = not all(foundational_status.values())
            missing_novel = not all(novel_status.values())
            
            # Check if we're using foundational models from config
            use_novel = self.config.get("models.use_novel", True)
            
            # Start download threads for missing models based on what we're using
            if missing_foundational and not use_novel:
                logger.info("Downloading missing foundational models")
                self.download_models("foundational", force=False, block=False)
                    
            if missing_novel and use_novel:
                logger.info("Downloading missing novel models")
                self.download_models("novel", force=False, block=False)
                    
        return all_status
    
    
    
    
    
    
    def wait_for_downloads(self):
        """
        Wait for all download threads to complete.
        
        Returns:
            bool: True if all downloads succeeded, False otherwise
        """
        for model_type, thread in self.download_threads.items():
            if thread.is_alive():
                logger.info(f"Waiting for {model_type} models to download...")
                thread.join()
        
        # Re-check availability after downloads
        foundational_status = self.check_model_availability("foundational")
        novel_status = self.check_model_availability("novel")
        
        # Check if all models are available
        all_available = all(foundational_status.values()) and all(novel_status.values())
        
        return all_available

# Command-line interface for the model initializer
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Initialize models for medical image enhancement application")
    parser.add_argument("--download", action="store_true", help="Download missing models")
    parser.add_argument("--force", action="store_true", help="Force re-download of all models")
    parser.add_argument("--type", choices=["foundational", "novel", "all"], default="all", help="Type of models to initialize")
    
    args = parser.parse_args()
    
    # Create initializer
    initializer = ModelInitializer()
    
    # Determine model type
    model_type = None if args.type == "all" else args.type
    
    # Check model availability
    availability = initializer.check_model_availability(model_type)
    
    # Print availability status
    print("Model availability:")
    for model_id, available in availability.items():
        print(f"  {model_id}: {'Available' if available else 'Not available'}")
    
    # Download models if requested
    if args.download or args.force:
        print(f"Downloading {'all' if model_type is None else model_type} models...")
        initializer.download_models(model_type, force=args.force, block=True)
        
        # Re-check availability
        availability = initializer.check_model_availability(model_type)
        
        # Print updated status
        print("Updated model availability:")
        for model_id, available in availability.items():
            print(f"  {model_id}: {'Available' if available else 'Not available'}")