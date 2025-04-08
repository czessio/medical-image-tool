#!/usr/bin/env python3
"""
Script to initialize and test the foundational models with pre-trained weights.
This helps verify that the weights are properly loaded and the models are working.
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import logging
from PIL import Image

# Add parent directory to path to access utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logging_setup import setup_logging
from utils.model_manager import ModelManager
from utils.model_downloader import ModelDownloader
from utils.weight_adapters import adapt_dncnn_weights
from ai.model_registry import ModelRegistry
from ai.cleaning.inference.cleaning_pipeline import CleaningPipeline

def create_test_image(size=(256, 256)):
    """Create a simple test image."""
    # Create a gradient image
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    
    # Create a more interesting pattern
    image = np.sin(xx * 10) * np.sin(yy * 10) * 0.5 + 0.5
    
    # Add some noise
    noise = np.random.normal(0, 0.05, size)
    image = np.clip(image + noise, 0, 1)
    
    # Convert to RGB
    image_rgb = np.stack([image] * 3, axis=2)
    
    return image_rgb

def test_models(args):
    """Test loading and inference with foundational models."""
    logger = logging.getLogger(__name__)
    logger.info("Testing foundational models...")
    
    # Create model manager
    model_manager = ModelManager()
    
    # Get available models
    foundational_models = model_manager.get_available_models(category="foundational", downloaded_only=True)
    
    if not foundational_models:
        logger.warning("No foundational models found. Please check weight files.")
        return

    logger.info(f"Found {len(foundational_models)} foundational models.")
    
    # Test each model individually
    for model_info in foundational_models:
        model_id = model_info["id"]
        logger.info(f"Testing model: {model_id}")
        
        # Skip if no image path provided and not running full tests
        if not args.image and not args.full:
            logger.info(f"Skipping inference test for {model_id} (no image provided)")
            continue
        
        # Create or load test image
        if args.image:
            logger.info(f"Loading test image: {args.image}")
            try:
                img = Image.open(args.image)
                test_img = np.array(img).astype(np.float32) / 255.0
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                test_img = create_test_image()
        else:
            logger.info("Creating synthetic test image")
            test_img = create_test_image()
        
        # Get the model
        model = model_manager.get_model(model_id)
        if model is None:
            logger.error(f"Failed to load model: {model_id}")
            continue
        
        # Run inference
        try:
            logger.info(f"Running inference with {model_id}...")
            result = model.process(test_img)
            
            logger.info(f"Inference successful. Result shape: {result.shape}")
            
            # Save result for visualization
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(exist_ok=True)
                
                # Convert to uint8 image
                result_img = (result * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(result_img)
                
                output_path = output_dir / f"{model_id}_result.png"
                pil_img.save(output_path)
                logger.info(f"Saved result to: {output_path}")
        
        except Exception as e:
            logger.error(f"Error during inference with {model_id}: {e}")
    
    # Test the cleaning pipeline
    if args.pipeline:
        logger.info("Testing cleaning pipeline with foundational models...")
        
        # Create pipeline
        pipeline = CleaningPipeline(use_novel_models=False)
        
        # Process the test image
        try:
            result = pipeline.process(test_img)
            
            logger.info(f"Pipeline processing successful. Result shape: {result.shape}")
            
            # Save result
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(exist_ok=True)
                
                # Convert to uint8 image
                result_img = (result * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(result_img)
                
                output_path = output_dir / "pipeline_result.png"
                pil_img.save(output_path)
                logger.info(f"Saved pipeline result to: {output_path}")
        
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Initialize and test foundational models")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--output-dir", default="output", help="Directory to save results")
    parser.add_argument("--pipeline", action="store_true", help="Test the cleaning pipeline")
    parser.add_argument("--full", action="store_true", help="Run full tests including inference")
    parser.add_argument("--log-level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level=log_level)
    
    # Test models
    test_models(args)

if __name__ == "__main__":
    main()