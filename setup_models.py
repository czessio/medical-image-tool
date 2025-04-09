#!/usr/bin/env python3
"""
Unified integration script for medical image enhancement models.
This script sets up the directory structure and copies your model files
to the right locations.
"""
import os
import sys
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_weights_directory():
    """Create the weights directory structure."""
    root = Path('.')
    
    # Create main directories
    weights_dir = root / 'weights'
    foundational_dir = weights_dir / 'foundational'
    
    weights_dir.mkdir(exist_ok=True)
    foundational_dir.mkdir(exist_ok=True)
    
    # Create model type subdirectories
    (foundational_dir / 'denoising').mkdir(exist_ok=True)
    (foundational_dir / 'super_resolution').mkdir(exist_ok=True)
    (foundational_dir / 'artifact_removal').mkdir(exist_ok=True)
    
    return foundational_dir

def copy_model_files(model_dir=None):
    """Copy model files to appropriate locations or update registry if files exist."""
    foundational_dir = setup_weights_directory()
    
    # Define target paths to check if they exist
    denoising_target = foundational_dir / 'denoising' / 'dncnn_25.pth'
    sr_target_x2 = foundational_dir / 'super_resolution' / 'RealESRGAN_x2.pth'
    sr_target_x4 = foundational_dir / 'super_resolution' / 'RealESRGAN_x4.pth'
    sr_target_x8 = foundational_dir / 'super_resolution' / 'RealESRGAN_x8.pth'
    artifact_target = foundational_dir / 'artifact_removal' / 'G_ema_ep_82.pth'
    
    # Check which files exist
    logger.info("Checking for existing model files...")
    if denoising_target.exists():
        logger.info(f"Found denoising model: {denoising_target}")
    if sr_target_x2.exists():
        logger.info(f"Found super-resolution model (x2): {sr_target_x2}")
    if sr_target_x4.exists():
        logger.info(f"Found super-resolution model (x4): {sr_target_x4}")
    if sr_target_x8.exists():
        logger.info(f"Found super-resolution model (x8): {sr_target_x8}")
    if artifact_target.exists():
        logger.info(f"Found artifact removal model: {artifact_target}")
    
    # If model directory is provided and files are missing, copy them
    if model_dir and Path(model_dir).exists():
        src_dir = Path(model_dir)
        
        # Only copy files that don't exist yet
        if not denoising_target.exists():
            # Look for denoising model files
            for file in src_dir.glob('**/*'):
                if file.is_file() and ('dncnn' in file.name.lower() or 'dncnn_25.pth' == file.name):
                    logger.info(f"Copying {file} to {denoising_target}")
                    shutil.copy2(file, denoising_target)
                    break
        
        if not sr_target_x2.exists():
            # Look for super-resolution x2 model files
            for file in src_dir.glob('**/*'):
                if file.is_file() and (('realsr' in file.name.lower() and 'x2' in file.name.lower()) or 
                                      'realesrgan_x2.pth' == file.name.lower()):
                    logger.info(f"Copying {file} to {sr_target_x2}")
                    shutil.copy2(file, sr_target_x2)
                    break
        
        if not sr_target_x4.exists():
            # Look for super-resolution x4 model files
            for file in src_dir.glob('**/*'):
                if file.is_file() and (('realsr' in file.name.lower() and 'x4' in file.name.lower()) or 
                                      'realesrgan_x4.pth' == file.name.lower()):
                    logger.info(f"Copying {file} to {sr_target_x4}")
                    shutil.copy2(file, sr_target_x4)
                    break
        
        if not sr_target_x8.exists():
            # Look for super-resolution x8 model files
            for file in src_dir.glob('**/*'):
                if file.is_file() and (('realsr' in file.name.lower() and 'x8' in file.name.lower()) or 
                                      'realesrgan_x8.pth' == file.name.lower()):
                    logger.info(f"Copying {file} to {sr_target_x8}")
                    shutil.copy2(file, sr_target_x8)
                    break
        
        if not artifact_target.exists():
            # Look for artifact removal model files
            for file in src_dir.glob('**/*'):
                if file.is_file() and ('g_ema' in file.name.lower() or 'artifact' in file.name.lower()):
                    logger.info(f"Copying {file} to {artifact_target}")
                    shutil.copy2(file, artifact_target)
                    break
    
    # Update model registry to point to the correct files
    update_model_registry()
    
    logger.info("Model integration complete!")
    logger.info("To use foundational models, run: python main.py --use-foundational")

def update_model_registry():
    """Update the model registry to point to the right files."""
    registry_file = Path('weights/model_registry.json')
    
    
    if not registry_file.exists():
        # Create default registry
        registry_content = '''{
    "dncnn_denoiser": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth",
        "md5": null,
        "file_name": "denoising/dncnn_25.pth",
        "description": "DnCNN model for grayscale blind denoising",
        "category": "foundational"
    },
    "edsr_super_resolution": {
        "url": "https://github.com/sanghyun-son/EDSR-PyTorch/raw/master/experiment/model/edsr_baseline_x2-1bc95232.pt",
        "md5": null,
        "file_name": "super_resolution/RealESRGAN_x2.pth",
        "description": "RealESRGAN model for 2x super-resolution",
        "category": "foundational"
    },
    "unet_artifact_removal": {
        "url": "https://github.com/mameli/Artifact_Removal_GAN",
        "md5": null,
        "file_name": "artifact_removal/G_ema_ep_82.pth",
        "description": "U-Net GAN model for artifact removal",
        "category": "foundational"
    }
}'''
        
        with open(registry_file, 'w') as f:
            f.write(registry_content)
            
        logger.info(f"Created model registry at {registry_file}")

if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    copy_model_files(model_dir)