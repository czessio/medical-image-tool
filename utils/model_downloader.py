"""
Model downloader utility for medical image enhancement application.
Handles downloading, caching, and verification of pre-trained model weights.
"""
import os
import sys
import urllib.request
import hashlib
import zipfile
import tarfile
import logging
import json
from pathlib import Path
import shutil
import time
from tqdm import tqdm  # For progress bars

from utils.config import Config

logger = logging.getLogger(__name__)

class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class ModelDownloader:
    """
    Utility for downloading and managing pre-trained model weights.
    
    Features:
    - Downloads pre-trained weights from specified URLs
    - Verifies downloads with checksum
    - Caches models to avoid re-downloading
    - Handles different archive formats (zip, tar.gz)
    """
    
    # Model registry - will be populated from a config file
    MODEL_REGISTRY = {
        # === Foundational Models ===
        "dncnn_denoiser": {
            "url": "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth",
            "md5": None,
            "file_name": "denoising/dncnn_gray_blind.pth",
            "description": "DnCNN model for grayscale blind denoising",
            "category": "foundational",
            "input_channels": 1,
            "output_channels": 1,
            "grayscale_only": True
        },
        "edsr_super_resolution": {
            "url": "https://github.com/sanghyun-son/EDSR-PyTorch/raw/master/experiment/model/edsr_baseline_x2-1bc95232.pt",
            "md5": None,
            "file_name": "super_resolution/edsr_x2.pt", 
            "description": "EDSR model for 2x super-resolution (baseline)",
            "category": "foundational",
            "scale_factor": 2,
            "input_channels": 3,
            "output_channels": 3,
            "grayscale_only": False
        },
        "unet_artifact_removal": {
            "url": "https://github.com/mameli/Artifact_Removal_GAN",
            "md5": None,
            "file_name": "artifact_removal/G_ema_ep_82.pth",
            "description": "U-Net GAN model for artifact removal",
            "category": "foundational",
            "input_channels": 1,
            "output_channels": 1,
            "grayscale_only": True
        },


        # === Novel Models ===
        "diffusion_denoiser": {
            "url": "https://github.com/yuanzhi-zhu/DiffPIR",
            "md5": None,
            "file_name": "DiffPIR_repo_link_only",
            "description": "DiffPIR: Denoising Diffusion Models for Plug-and-Play Restoration (manual download)",
            "category": "novel",
            "input_channels": 3,
            "output_channels": 3,
            "grayscale_only": False
        },
        "swinir_super_resolution": {
            "url": "https://github.com/JingyunLiang/SwinIR",
            "md5": None,
            "file_name": "SwinIR_repo_link_only",
            "description": "SwinIR official GitHub repository (manual download)",
            "category": "novel",
            "scale_factor": 2,
            "input_channels": 3,
            "output_channels": 3,
            "grayscale_only": False
        },
        "stylegan_artifact_removal": {
            "url": "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3",
            "md5": None,
            "file_name": "StyleGAN3_NGC_link_only",
            "description": "StyleGAN3 by NVIDIA Research (download via NGC CLI)",
            "category": "novel",
            "input_channels": 3,
            "output_channels": 3,
            "grayscale_only": False
        }
    }





    
    def __init__(self, base_dir=None, config=None):
        """
        Initialize the model downloader.
        
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
        
        # Ensure all directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.foundational_dir.mkdir(exist_ok=True)
        self.novel_dir.mkdir(exist_ok=True)
        
        # Create model registry file if it doesn't exist
        self.registry_file = self.base_dir / "model_registry.json"
        if not self.registry_file.exists():
            self._save_registry()
        else:
            self._load_registry()
    
    def _save_registry(self):
        """Save the model registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.MODEL_REGISTRY, f, indent=4)
                logger.info(f"Model registry saved to {self.registry_file}")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def _load_registry(self):
        """Load the model registry from file."""
        try:
            with open(self.registry_file, 'r') as f:
                loaded_registry = json.load(f)
                # Update with loaded registry but keep defaults as fallback
                for model_id, model_info in loaded_registry.items():
                    if model_id in self.MODEL_REGISTRY:
                        self.MODEL_REGISTRY[model_id].update(model_info)
                    else:
                        self.MODEL_REGISTRY[model_id] = model_info
                logger.info(f"Model registry loaded from {self.registry_file}")
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
            # Save the default registry if loading fails
            self._save_registry()
    
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
        Download a model by ID.
        
        Args:
            model_id: ID of the model to download
            force: Whether to force re-download even if the file exists
            
        Returns:
            Path: Path to the downloaded model file or None if download failed
        """
        if model_id not in self.MODEL_REGISTRY:
            logger.error(f"Model ID not found in registry: {model_id}")
            return None
        
        model_info = self.MODEL_REGISTRY[model_id]
        url = model_info.get("url")
        expected_md5 = model_info.get("md5")
        category = model_info.get("category", "foundational")
        file_name = model_info.get("file_name", f"{model_id}.pth")
        
        # Determine the directory based on category
        if category == "novel":
            model_dir = self.novel_dir
        else:
            model_dir = self.foundational_dir
            
        model_path = model_dir / file_name
        
        # Check if the file already exists and has correct checksum
        if model_path.exists() and not force:
            if self._verify_checksum(model_path, expected_md5):
                logger.info(f"Model already exists with correct checksum: {model_path}")
                return model_path
            else:
                logger.warning(f"Model exists but has incorrect checksum: {model_path}")
                # Continue to re-download
        
        # Download the file
        try:
            # Create a temporary file for download
            temp_file = model_dir / f"{file_name}.download"
            
            logger.info(f"Downloading model from {url} to {temp_file}")
            
            # Download with progress bar
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f"Downloading {model_id}") as t:
                urllib.request.urlretrieve(url, temp_file, reporthook=t.update_to)
            
            # Verify the download
            if expected_md5 and not self._verify_checksum(temp_file, expected_md5):
                logger.error(f"Download verification failed for {model_id}")
                temp_file.unlink(missing_ok=True)
                return None
            
            # Move the file to the final location
            shutil.move(temp_file, model_path)
            logger.info(f"Model downloaded successfully: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            return None
    
    def download_all_models(self, category=None, force=False):
        """
        Download all models in the registry or a specific category.
        
        Args:
            category: Category of models to download ("foundational", "novel", or None for all)
            force: Whether to force re-download even if files exist
            
        Returns:
            dict: Dictionary of model_id -> download results (True/False)
        """
        results = {}
        
        for model_id, model_info in self.MODEL_REGISTRY.items():
            # Skip if category doesn't match
            if category and model_info.get("category") != category:
                continue
                
            # Download the model
            path = self.download_model(model_id, force)
            results[model_id] = path is not None
        
        return results
    
    def _verify_checksum(self, file_path, expected_md5):
        """
        Verify a file's MD5 checksum.
        
        Args:
            file_path: Path to the file
            expected_md5: Expected MD5 checksum
            
        Returns:
            bool: True if the checksums match, False otherwise
        """
        if not expected_md5:
            # No checksum to verify, skip verification
            return True
            
        try:
            # Calculate the file's MD5
            md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5.update(chunk)
            actual_md5 = md5.hexdigest()
            
            # Compare with expected
            return actual_md5.lower() == expected_md5.lower()
            
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return False
    
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
    
    def update_model_registry(self, model_id, url, md5=None, file_name=None, description=None, category="foundational"):
        """
        Update or add a model to the registry.
        
        Args:
            model_id: ID of the model
            url: URL to download the model
            md5: MD5 checksum for verification
            file_name: Name to save the file as
            description: Description of the model
            category: Category of the model ("foundational" or "novel")
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create or update model entry
        if model_id not in self.MODEL_REGISTRY:
            self.MODEL_REGISTRY[model_id] = {}
            
        model_info = self.MODEL_REGISTRY[model_id]
        model_info["url"] = url
        
        if md5:
            model_info["md5"] = md5
        if file_name:
            model_info["file_name"] = file_name
        if description:
            model_info["description"] = description
        if category:
            model_info["category"] = category
            
        # Save the updated registry
        self._save_registry()
        return True

# Command-line interface for the model downloader
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Medical image enhancement model downloader")
    parser.add_argument("action", choices=["download", "list", "verify"], help="Action to perform")
    parser.add_argument("--model", help="Model ID (or 'all' for all models)")
    parser.add_argument("--category", choices=["foundational", "novel"], help="Model category")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create downloader
    downloader = ModelDownloader()
    
    # Handle actions
    if args.action == "download":
        if args.model == "all":
            results = downloader.download_all_models(category=args.category, force=args.force)
            for model_id, success in results.items():
                print(f"{model_id}: {'Downloaded' if success else 'Failed'}")
        else:
            if not args.model:
                parser.error("--model is required for download action")
            path = downloader.download_model(args.model, force=args.force)
            if path:
                print(f"Model downloaded to: {path}")
            else:
                print(f"Failed to download model: {args.model}")
                sys.exit(1)
    
    elif args.action == "list":
        models = downloader.list_available_models(category=args.category)
        print(f"Available models ({len(models)}):")
        for model in models:
            status = "Downloaded" if model["downloaded"] else "Not downloaded"
            print(f"- {model['id']} ({model['category']}): {model['description']} [{status}]")
    
    elif args.action == "verify":
        if args.model == "all":
            models = downloader.list_available_models(category=args.category, downloaded_only=True)
            for model in models:
                model_id = model["id"]
                model_info = downloader.MODEL_REGISTRY[model_id]
                path = Path(model["path"])
                md5 = model_info.get("md5")
                if md5:
                    valid = downloader._verify_checksum(path, md5)
                    print(f"{model_id}: {'Valid' if valid else 'Invalid'}")
                else:
                    print(f"{model_id}: No checksum available")
        else:
            if not args.model:
                parser.error("--model is required for verify action")
            model_info = downloader.MODEL_REGISTRY.get(args.model)
            if not model_info:
                print(f"Model not found in registry: {args.model}")
                sys.exit(1)
            path = downloader.get_model_path(args.model)
            if not path:
                print(f"Model not downloaded: {args.model}")
                sys.exit(1)
            md5 = model_info.get("md5")
            if not md5:
                print(f"No checksum available for model: {args.model}")
                sys.exit(1)
            valid = downloader._verify_checksum(path, md5)
            print(f"Checksum {'valid' if valid else 'invalid'} for {args.model}")
            if not valid:
                sys.exit(1)