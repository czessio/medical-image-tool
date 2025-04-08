#!/usr/bin/env python3
"""
Script to clean up the model weights directory.
This removes temporary files and can reset the directory structure.
"""
import os
import sys
import argparse
import shutil
from pathlib import Path
import logging

# Add parent directory to path to access utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logging_setup import setup_logging

def clean_weights_directory(args):
    """
    Clean up the weights directory.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent.parent
    weights_dir = root_dir / "weights"
    
    if not weights_dir.exists():
        logger.warning(f"Weights directory not found: {weights_dir}")
        return
    
    logger.info(f"Cleaning weights directory: {weights_dir}")
    
    # Remove temporary files
    if args.temp:
        logger.info("Removing temporary files...")
        for ext in ['.download', '.tmp', '.bak']:
            for temp_file in weights_dir.glob(f"**/*{ext}"):
                logger.info(f"Removing temporary file: {temp_file}")
                temp_file.unlink()
    
    # Reset directory structure
    if args.reset:
        logger.warning("Resetting weights directory structure...")
        
        # Ask for confirmation if keep_files is False
        if not args.keep_files:
            if not args.yes:
                confirm = input("This will delete all model weight files! Are you sure? (y/N): ")
                if confirm.lower() != 'y':
                    logger.info("Operation cancelled.")
                    return
        
        # Remove all files and subdirectories
        for item in weights_dir.glob("*"):
            if item.is_dir():
                if args.keep_files:
                    # Keep weight files but reset directory structure
                    for file in item.glob("**/*.pth") + item.glob("**/*.pt"):
                        # Move to a temp location
                        temp_file = weights_dir / file.name
                        logger.info(f"Preserving file: {file} -> {temp_file}")
                        shutil.move(str(file), str(temp_file))
                
                logger.info(f"Removing directory: {item}")
                shutil.rmtree(item)
            elif item.name != "README.md" and not args.keep_files:
                logger.info(f"Removing file: {item}")
                item.unlink()
        
        # Recreate directory structure
        logger.info("Recreating directory structure...")
        setup_cmd = f"python {root_dir / 'scripts' / 'setup_weights_dir.py'}"
        os.system(setup_cmd)
        
        # Move back any preserved files
        if args.keep_files:
            for file in weights_dir.glob("*.pth") + weights_dir.glob("*.pt"):
                # Determine appropriate destination
                if "dncnn" in file.name.lower():
                    dest_dir = weights_dir / "foundational" / "denoising"
                elif "edsr" in file.name.lower() or "realsr" in file.name.lower() or "esrgan" in file.name.lower():
                    dest_dir = weights_dir / "foundational" / "super_resolution"
                elif "unet" in file.name.lower() or "gan" in file.name.lower() or "g_ema" in file.name.lower():
                    dest_dir = weights_dir / "foundational" / "artifact_removal"
                else:
                    # If can't determine, keep in root
                    continue
                
                dest_file = dest_dir / file.name
                logger.info(f"Moving file to appropriate directory: {file} -> {dest_file}")
                shutil.move(str(file), str(dest_file))
    
    logger.info("Weights directory cleaned up successfully!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up weights directory")
    parser.add_argument("--temp", action="store_true", 
                      help="Remove temporary files")
    parser.add_argument("--reset", action="store_true", 
                      help="Reset directory structure")
    parser.add_argument("--keep-files", action="store_true", 
                      help="Keep weight files when resetting")
    parser.add_argument("--yes", "-y", action="store_true", 
                      help="Don't ask for confirmation")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Default to removing temp files if no other action specified
    if not (args.temp or args.reset):
        args.temp = True
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level=log_level)
    
    # Clean weights directory
    clean_weights_directory(args)

if __name__ == "__main__":
    main()