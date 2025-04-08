#!/usr/bin/env python3
"""
Simple model integration script for foundational models.
"""
import os
import sys
import shutil
from pathlib import Path
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_weight_directories():
    """Create the necessary directories for model weights."""
    # Create weight directories
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    foundational_dir = weights_dir / 'foundational'
    foundational_dir.mkdir(exist_ok=True)
    
    # Create model type directories
    (foundational_dir / 'denoising').mkdir(exist_ok=True)
    (foundational_dir / 'super_resolution').mkdir(exist_ok=True)
    (foundational_dir / 'artifact_removal').mkdir(exist_ok=True)
    
    logger.info("Created weight directories")
    
    return foundational_dir

def integrate_model_weights(source_dir=None):
    """
    Integrate model weights into the application.
    
    Args:
        source_dir: Directory containing model weights (optional)
    """
    # Setup directories
    foundational_dir = setup_weight_directories()
    
    # If a source directory is provided, copy weight files
    if source_dir and Path(source_dir).exists():
        source_path = Path(source_dir)
        
        # Look for weight files in the source directory
        for file in source_path.glob('*.pth'):
            # Determine the appropriate destination
            if "dncnn" in file.name.lower():
                dest_dir = foundational_dir / "denoising"
                dest_file = dest_dir / "dncnn_gray_blind.pth"
            elif "edsr" in file.name.lower() or "esrgan" in file.name.lower():
                dest_dir = foundational_dir / "super_resolution"
                dest_file = dest_dir / ("edsr_x2.pt" if "edsr" in file.name.lower() else "RealESRGAN_x4.pth")
            elif "unet" in file.name.lower() or "gan" in file.name.lower():
                dest_dir = foundational_dir / "artifact_removal"
                dest_file = dest_dir / "G_ema_ep_82.pth"
            else:
                logger.warning(f"Unknown model type for {file}, skipping")
                continue
            
            # Copy the file
            logger.info(f"Copying {file} to {dest_file}")
            shutil.copy(file, dest_file)
    
    # Create simple metadata files if they don't exist
    create_metadata_files(foundational_dir)
    
    logger.info("Model integration complete")
    print("\nTo use foundational models, run the application with:")
    print("python main.py --use-foundational")

def create_metadata_files(foundational_dir):
    """Create basic metadata files for each model type."""
    # DnCNN metadata
    dncnn_meta = foundational_dir / "denoising" / "metadata.json"
    if not dncnn_meta.exists():
        with open(dncnn_meta, 'w') as f:
            f.write('{\n')
            f.write('  "model_name": "DnCNN",\n')
            f.write('  "file": "dncnn_gray_blind.pth",\n')
            f.write('  "task": "denoising",\n')
            f.write('  "input_channels": 1,\n')
            f.write('  "output_channels": 1\n')
            f.write('}\n')
    
    # EDSR metadata
    edsr_meta = foundational_dir / "super_resolution" / "metadata.json"
    if not edsr_meta.exists():
        with open(edsr_meta, 'w') as f:
            f.write('{\n')
            f.write('  "model_name": "EDSR",\n')
            f.write('  "file": "edsr_x2.pt",\n')
            f.write('  "task": "super_resolution",\n')
            f.write('  "input_channels": 3,\n')
            f.write('  "output_channels": 3,\n')
            f.write('  "scale_factor": 2\n')
            f.write('}\n')
    
    # U-Net GAN metadata
    unet_meta = foundational_dir / "artifact_removal" / "metadata.json"
    if not unet_meta.exists():
        with open(unet_meta, 'w') as f:
            f.write('{\n')
            f.write('  "model_name": "UNetGAN",\n')
            f.write('  "file": "G_ema_ep_82.pth",\n')
            f.write('  "task": "artifact_removal",\n')
            f.write('  "input_channels": 3,\n')
            f.write('  "output_channels": 3\n')
            f.write('}\n')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If a source directory is provided
        integrate_model_weights(sys.argv[1])
    else:
        integrate_model_weights()