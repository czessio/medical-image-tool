#!/usr/bin/env python3
"""
Script to set up the model weights directory structure.
This creates the necessary directories and copies metadata files.
"""
import os
import sys
import shutil
import json
from pathlib import Path
import logging

# Add parent directory to path to access utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logging_setup import setup_logging

def setup_directories():
    """Set up the model weights directory structure."""
    logger = logging.getLogger(__name__)
    logger.info("Setting up model weights directory structure...")
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent.parent
    
    # Create directory structure
    weights_dir = root_dir / "weights"
    foundational_dir = weights_dir / "foundational"
    novel_dir = weights_dir / "novel"
    
    # Create foundational model category directories
    denoising_dir = foundational_dir / "denoising"
    super_res_dir = foundational_dir / "super_resolution"
    artifact_dir = foundational_dir / "artifact_removal"
    
    # Create all directories
    for directory in [weights_dir, foundational_dir, novel_dir, 
                     denoising_dir, super_res_dir, artifact_dir]:
        directory.mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Copy metadata files if they exist in the project
    metadata_sources = {
        "foundational/denoising/metadata.json": denoising_dir / "metadata.json",
        "foundational/super_resolution/metadata.json": super_res_dir / "metadata.json",
        "foundational/artifact_removal/metadata.json": artifact_dir / "metadata.json"
    }
    
    for src_path, dest_path in metadata_sources.items():
        src_full_path = root_dir / src_path
        if src_full_path.exists() and not dest_path.exists():
            shutil.copy(src_full_path, dest_path)
            logger.info(f"Copied metadata: {src_path} -> {dest_path}")
    
    # Create registry file if it doesn't exist
    registry_file = weights_dir / "model_registry.json"
    if not registry_file.exists():
        # Create basic registry skeleton
        registry = {
            "dncnn_denoiser": {
                "url": "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth",
                "md5": None,
                "file_name": "denoising/dncnn_gray_blind.pth",
                "description": "DnCNN model for grayscale blind denoising",
                "category": "foundational"
            },
            "edsr_super_resolution": {
                "url": "https://github.com/sanghyun-son/EDSR-PyTorch/raw/master/experiment/model/edsr_baseline_x2-1bc95232.pt",
                "md5": None,
                "file_name": "super_resolution/edsr_x2.pt",
                "description": "EDSR model for 2x super-resolution (baseline)",
                "category": "foundational"
            },
            "unet_artifact_removal": {
                "url": "https://github.com/mameli/Artifact_Removal_GAN",
                "md5": None,
                "file_name": "artifact_removal/G_ema_ep_82.pth",
                "description": "U-Net GAN model for artifact removal",
                "category": "foundational"
            }
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=4)
        
        logger.info(f"Created model registry file: {registry_file}")
    
    logger.info("Directory setup complete.")
    return weights_dir

def main():
    """Main function."""
    # Set up logging
    logger = setup_logging()
    
    # Set up directories
    weights_dir = setup_directories()
    
    # Print instructions
    print("\nModel weights directory structure set up successfully!")
    print(f"Weight files should be placed in: {weights_dir}")
    print("\nRequired weight files:")
    print("  weights/foundational/denoising/dncnn_gray_blind.pth")
    print("  weights/foundational/super_resolution/edsr_x2.pt")
    print("  weights/foundational/artifact_removal/G_ema_ep_82.pth")
    print("\nRun the adapt_weights.py script to adapt weight files if needed.")

if __name__ == "__main__":
    main()