#!/usr/bin/env python3
"""
Script to integrate pre-trained model weights into the application.
This sets up the directory structure, adapts weights if needed, and tests the models.
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
from utils.weight_adapters import adapt_dncnn_weights, analyze_realergan_weights

def integrate_models(args):
    """
    Integrate pre-trained model weights into the application.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info("Integrating pre-trained model weights...")
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent.parent
    
    # First, set up the directory structure
    setup_cmd = f"python {root_dir / 'scripts' / 'setup_weights_dir.py'}"
    logger.info(f"Setting up directory structure: {setup_cmd}")
    os.system(setup_cmd)
    
    # Adapt weights if source paths are provided
    adapt_needed = False
    adapt_cmd = f"python {root_dir / 'scripts' / 'adapt_weights.py'}"
    
    if args.dncnn:
        adapt_cmd += f" --dncnn {args.dncnn}"
        adapt_needed = True
    
    if args.edsr:
        adapt_cmd += f" --edsr {args.edsr}"
        adapt_needed = True
    
    if args.realsr:
        adapt_cmd += f" --realsr {args.realsr}"
        adapt_needed = True
        
    if args.unet:
        adapt_cmd += f" --unet {args.unet}"
        adapt_needed = True
    
    if adapt_needed:
        logger.info(f"Adapting weights: {adapt_cmd}")
        os.system(adapt_cmd)
    else:
        logger.info("No weight sources provided for adaptation")
    
    # Test the models
    if args.test:
        test_cmd = f"python {root_dir / 'scripts' / 'initialize_models.py'} --full"
        if args.image:
            test_cmd += f" --image {args.image}"
        logger.info(f"Testing models: {test_cmd}")
        os.system(test_cmd)
    
    logger.info("Integration complete!")
    
    # Print usage instructions
    print("\nModel weights integration completed!")
    print("To use these models in the application:")
    print("1. Make sure the weights are properly placed in the 'weights/foundational/' directory")
    print("2. Run the application with the --use-foundational flag:")
    print("   python main.py --use-foundational")
    print("3. Or set 'use_novel' to False in the configuration")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Integrate pre-trained model weights")
    parser.add_argument("--dncnn", help="Path to DnCNN weights source")
    parser.add_argument("--edsr", help="Path to EDSR weights source")
    parser.add_argument("--realsr", help="Path to RealESRGAN weights source (alternative to EDSR)")
    parser.add_argument("--unet", help="Path to U-Net GAN weights source")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--no-test", dest="test", action="store_false", 
                        help="Skip model testing")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.set_defaults(test=True)
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level=log_level)
    
    # Integrate models
    integrate_models(args)

if __name__ == "__main__":
    main()