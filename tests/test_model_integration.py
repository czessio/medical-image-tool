#!/usr/bin/env python3
"""
Test script for model integration.
Can be used in CI/CD pipelines to ensure models are properly integrated.
"""
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logging_setup import setup_logging
from utils.model_manager import ModelManager
from ai.cleaning.inference.cleaning_pipeline import CleaningPipeline

# Set up logging
logger = setup_logging(log_level=logging.INFO)

class TestModelIntegration(unittest.TestCase):
    """Test cases for model integration."""

    def setUp(self):
        """Set up test environment."""
        self.model_manager = ModelManager()
        
        # Create a simple test image
        size = (128, 128)
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        self.test_image = np.sin(xx * 10) * np.sin(yy * 10) * 0.5 + 0.5
        self.test_image_rgb = np.stack([self.test_image] * 3, axis=2)

    def test_model_registration(self):
        """Test that models are registered correctly."""
        from ai.model_registry import ModelRegistry
        
        # Check for foundational models
        model_types = ModelRegistry._registry.keys()
        self.assertIn("dncnn_denoiser", model_types, "DnCNN model not registered")
        self.assertIn("edsr_super_resolution", model_types, "EDSR model not registered")
        self.assertIn("unet_artifact_removal", model_types, "U-Net GAN model not registered")

    def test_model_availability(self):
        """Test model availability."""
        # Get available foundational models
        foundational_models = self.model_manager.get_available_models(
            category="foundational", downloaded_only=True)
        
        # Check if we have at least one foundational model
        self.assertTrue(len(foundational_models) > 0, 
                       "No foundational models found. Integration may not be complete.")

    def test_model_loading(self):
        """Test that models can be loaded."""
        # Get available models
        available_models = self.model_manager.get_available_models(downloaded_only=True)
        
        if not available_models:
            self.skipTest("No models available for testing")
        
        # Try to load each available model
        for model_info in available_models:
            model_id = model_info["id"]
            model = self.model_manager.get_model(model_id)
            self.assertIsNotNone(model, f"Failed to load model: {model_id}")

    def test_cleaning_pipeline(self):
        """Test the cleaning pipeline with foundational models."""
        # Create pipeline with foundational models
        pipeline = CleaningPipeline(use_novel_models=False)
        
        # Check if any models were loaded
        active_models = pipeline.get_active_models()
        if not active_models:
            self.skipTest("No foundational models available in the pipeline")
        
        # Process a test image
        result = pipeline.process(self.test_image_rgb)
        
        # Check that the result has the same shape as the input
        self.assertEqual(result.shape, self.test_image_rgb.shape, 
                        "Output image has different shape than input")
        
        # Check that the result is not identical to the input
        # This is a bit tricky because even with no effective processing, floating point
        # operations can cause small differences. We'll use a tolerance.
        if len(active_models) > 0:
            differences = np.abs(result - self.test_image_rgb)
            has_differences = np.max(differences) > 1e-6
            self.assertTrue(has_differences, 
                           "Output is identical to input despite active models")

    def test_individual_models(self):
        """Test each individual model if available."""
        # Test DnCNN
        dncnn = self.model_manager.get_model("dncnn_denoiser")
        if dncnn is not None:
            # Add noise to the test image
            noisy_image = self.test_image_rgb + np.random.normal(0, 0.1, self.test_image_rgb.shape)
            noisy_image = np.clip(noisy_image, 0, 1)
            
            # Process with DnCNN
            denoised = dncnn.process(noisy_image)
            
            # Check that the result has the same shape
            self.assertEqual(denoised.shape, noisy_image.shape, 
                            "DnCNN output has different shape than input")
            
            # Check that some denoising occurred (MSE with clean should be less than MSE with noisy)
            mse_noisy = np.mean((self.test_image_rgb - noisy_image) ** 2)
            mse_denoised = np.mean((self.test_image_rgb - denoised) ** 2)
            self.assertLess(mse_denoised, mse_noisy, 
                           "DnCNN denoising did not improve image quality")
        
        # Test EDSR
        edsr = self.model_manager.get_model("edsr_super_resolution")
        if edsr is not None:
            # Downsample the test image
            h, w = self.test_image_rgb.shape[:2]
            downsampled = self.test_image_rgb[::2, ::2, :]
            
            # Process with EDSR
            upscaled = edsr.process(downsampled)
            
            # Check that the result has been upscaled
            self.assertGreater(upscaled.shape[0], downsampled.shape[0], 
                              "EDSR did not increase image height")
            self.assertGreater(upscaled.shape[1], downsampled.shape[1], 
                              "EDSR did not increase image width")
        
        # Test U-Net GAN
        unet = self.model_manager.get_model("unet_artifact_removal")
        if unet is not None:
            # Add artifacts to the test image
            artifacted_image = self.test_image_rgb.copy()
            h, w = artifacted_image.shape[:2]
            # Add some line artifacts
            for i in range(0, h, 8):
                artifacted_image[i:i+2, :, :] = 1.0
            
            # Process with U-Net GAN
            cleaned = unet.process(artifacted_image)
            
            # Check that the result has the same shape
            self.assertEqual(cleaned.shape, artifacted_image.shape, 
                            "U-Net GAN output has different shape than input")
            
            # Check that artifact removal occurred by comparing artifacts still present
            artifacts_before = np.sum(artifacted_image[::8, :, :] == 1.0)
            artifacts_after = np.sum(cleaned[::8, :, :] == 1.0)
            self.assertLess(artifacts_after, artifacts_before, 
                           "U-Net GAN did not remove artifacts")

if __name__ == "__main__":
    unittest.main()