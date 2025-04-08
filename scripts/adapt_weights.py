#!/usr/bin/env python3
"""
Script to adapt model weights to the format expected by the application.
This handles any necessary conversions between weight formats.
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import logging

# Add parent directory to path to access utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.weight_adapters import adapt_dncnn_weights, analyze_realergan_weights
from utils.logging_setup import setup_logging

def main():
    """Main function for weight adaptation."""
    parser = argparse.ArgumentParser(description="Adapt model weights for the application")
    parser.add_argument("--dncnn", help="Path to DnCNN weights to adapt")
    parser.add_argument("--edsr", help="Path to EDSR weights to adapt")
    parser.add_argument("--realsr", help="Path to RealESRGAN weights to adapt")
    parser.add_argument("--unet", help="Path to UNet-GAN weights to adapt")
    parser.add_argument("--output-dir", default="weights/foundational", 
                        help="Output directory for adapted weights")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Process DnCNN weights
    if args.dncnn:
        logger.info(f"Adapting DnCNN weights: {args.dncnn}")
        dncnn_out_dir = os.path.join(args.output_dir, "denoising")
        os.makedirs(dncnn_out_dir, exist_ok=True)
        
        output_path = os.path.join(dncnn_out_dir, "dncnn_gray_blind.pth")
        adapt_dncnn_weights(args.dncnn, output_path)
        logger.info(f"DnCNN weights adapted and saved to: {output_path}")
    
    # Process EDSR weights
    if args.edsr:
        logger.info(f"Adapting EDSR weights: {args.edsr}")
        edsr_out_dir = os.path.join(args.output_dir, "super_resolution")
        os.makedirs(edsr_out_dir, exist_ok=True)
        
        output_path = os.path.join(edsr_out_dir, "edsr_x2.pt")
        # Simply copy for now, as EDSR weights might not need adaptation
        try:
            state_dict = torch.load(args.edsr, map_location='cpu')
            torch.save(state_dict, output_path)
            logger.info(f"EDSR weights saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error adapting EDSR weights: {e}")
    
    # Process RealESRGAN weights
    if args.realsr:
        logger.info(f"Analyzing RealESRGAN weights: {args.realsr}")
        realsr_out_dir = os.path.join(args.output_dir, "super_resolution")
        os.makedirs(realsr_out_dir, exist_ok=True)
        
        # Analyze weights to understand their structure
        analysis = analyze_realergan_weights(args.realsr)
        
        # Copy weights to destination
        output_path = os.path.join(realsr_out_dir, "RealESRGAN_x4.pth")
        try:
            state_dict = torch.load(args.realsr, map_location='cpu')
            
            # Extract params_ema if present (typical RealESRGAN format)
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
                
            torch.save(state_dict, output_path)
            logger.info(f"RealESRGAN weights saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error adapting RealESRGAN weights: {e}")
    
    # Process UNet-GAN weights
    if args.unet:
        logger.info(f"Processing UNet-GAN weights: {args.unet}")
        unet_out_dir = os.path.join(args.output_dir, "artifact_removal")
        os.makedirs(unet_out_dir, exist_ok=True)
        
        output_path = os.path.join(unet_out_dir, "G_ema_ep_82.pth")
        try:
            state_dict = torch.load(args.unet, map_location='cpu')
            
            # If this is a GAN with separate G/D models, extract the generator
            if isinstance(state_dict, dict) and 'G' in state_dict:
                state_dict = state_dict['G']
                
            torch.save(state_dict, output_path)
            logger.info(f"UNet-GAN weights saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error processing UNet-GAN weights: {e}")

if __name__ == "__main__":
    main()