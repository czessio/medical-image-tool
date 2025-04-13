"""
Export functionality for the medical image enhancement application.
Handles exporting enhanced images in various formats with options.
"""
import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image_loader import ImageLoader

logger = logging.getLogger(__name__)

class Exporter:
    """Handles exporting of processed images in various formats."""
    
    @staticmethod
    def export_image(image_data, output_path, metadata=None, is_medical_format=False, **kwargs):
        """
        Export a processed image to the specified path.
        
        Args:
            image_data: Numpy array containing the image data
            output_path: Path to save the exported image
            metadata: Optional metadata dictionary
            is_medical_format: Whether to save as a medical format
            **kwargs: Additional export options
            
        Returns:
            bool: True if successful, False otherwise
        """
        return ImageLoader.save_image(
            image_data, output_path, metadata, is_medical_format
        )
    
    
    
    

    @staticmethod
    def create_comparison_image(original, enhanced, output_path=None, mode='side_by_side'):
        """
        Create a comparison image showing original and enhanced versions.
        
        Args:
            original: Original image data as numpy array
            enhanced: Enhanced image data as numpy array
            output_path: Optional path to save the comparison image
            mode: Comparison mode ('side_by_side', 'overlay', or 'split')
            
        Returns:
            numpy.ndarray: Comparison image data
            If output_path is provided, also saves the image and returns success boolean
        """
        try:
            # Ensure images are valid
            if original is None or enhanced is None:
                logger.error("Cannot create comparison: missing image data")
                return None
            
            # Check for non-finite values and fix if needed
            if not np.isfinite(original).all():
                original = np.nan_to_num(original, nan=0.0, posinf=1.0, neginf=0.0)
            if not np.isfinite(enhanced).all():
                enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0)
                
            # Ensure images are in the correct format
            if np.issubdtype(original.dtype, np.floating):
                original_display = (original * 255).clip(0, 255).astype(np.uint8)
            else:
                original_display = original.copy()
                
            if np.issubdtype(enhanced.dtype, np.floating):
                enhanced_display = (enhanced * 255).clip(0, 255).astype(np.uint8)
            else:
                enhanced_display = enhanced.copy()
            
            # Ensure both images have the same size
            if original_display.shape != enhanced_display.shape:
                logger.warning("Original and enhanced images have different shapes. Resizing original.")
                # Resize original to match enhanced
                from data.processing.transforms import resize_image
                
                # Handle different dimensionality
                if len(original_display.shape) != len(enhanced_display.shape):
                    if len(original_display.shape) == 2 and len(enhanced_display.shape) == 3:
                        # Convert grayscale to RGB
                        original_display = np.stack([original_display] * 3, axis=2)
                    elif len(original_display.shape) == 3 and len(enhanced_display.shape) == 2:
                        # Convert enhanced grayscale to RGB
                        enhanced_display = np.stack([enhanced_display] * 3, axis=2)
                
                # Now resize to match dimensions
                original_display = resize_image(
                    original_display, 
                    (enhanced_display.shape[1], enhanced_display.shape[0]),
                    preserve_aspect_ratio=False
                )
            
            # Create comparison based on mode
            if mode == 'side_by_side':
                # Create side-by-side comparison
                height = original_display.shape[0]
                width = original_display.shape[1]
                
                # Create a larger canvas for the two images
                comparison = np.zeros((height, width * 2 + 10, 3), dtype=np.uint8)
                
                # Handle different channel counts
                if len(original_display.shape) == 2:
                    original_rgb = np.stack([original_display] * 3, axis=2)
                else:
                    original_rgb = original_display
                    
                if len(enhanced_display.shape) == 2:
                    enhanced_rgb = np.stack([enhanced_display] * 3, axis=2)
                else:
                    enhanced_rgb = enhanced_display
                    
                # If we have 4 channels, use only the first 3
                if original_rgb.shape[2] > 3:
                    original_rgb = original_rgb[:, :, :3]
                if enhanced_rgb.shape[2] > 3:
                    enhanced_rgb = enhanced_rgb[:, :, :3]
                
                comparison[:, :width, :] = original_rgb
                comparison[:, width + 10:, :] = enhanced_rgb
                
                # Add labels
                comparison_pil = Image.fromarray(comparison)
                draw = ImageDraw.Draw(comparison_pil)
                
                # Try to get a font, if not available, just use basic drawing without text
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                
                if font:
                    draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
                    draw.text((width + 20, 10), "Enhanced", fill=(255, 255, 255), font=font)
                
                comparison = np.array(comparison_pil)
                
            elif mode == 'overlay':
                # Create a 50% blend of the two images
                # Make sure both images have the same number of channels
                if len(original_display.shape) != len(enhanced_display.shape):
                    if len(original_display.shape) == 2 and len(enhanced_display.shape) == 3:
                        original_display = np.stack([original_display] * 3, axis=2)
                    elif len(original_display.shape) == 3 and len(enhanced_display.shape) == 2:
                        enhanced_display = np.stack([enhanced_display] * 3, axis=2)
                
                # If we have 4 channels, use only the first 3
                if len(original_display.shape) == 3 and original_display.shape[2] > 3:
                    original_display = original_display[:, :, :3]
                if len(enhanced_display.shape) == 3 and enhanced_display.shape[2] > 3:
                    enhanced_display = enhanced_display[:, :, :3]
                
                comparison = (original_display.astype(np.float32) * 0.5 + enhanced_display.astype(np.float32) * 0.5).astype(np.uint8)
                
            elif mode == 'split':
                # Create a split view (left half is original, right half is enhanced)
                # Make sure both images have the same number of channels
                if len(original_display.shape) != len(enhanced_display.shape):
                    if len(original_display.shape) == 2 and len(enhanced_display.shape) == 3:
                        original_display = np.stack([original_display] * 3, axis=2)
                    elif len(original_display.shape) == 3 and len(enhanced_display.shape) == 2:
                        enhanced_display = np.stack([enhanced_display] * 3, axis=2)
                
                # If we have 4 channels, use only the first 3
                if len(original_display.shape) == 3 and original_display.shape[2] > 3:
                    original_display = original_display[:, :, :3]
                if len(enhanced_display.shape) == 3 and enhanced_display.shape[2] > 3:
                    enhanced_display = enhanced_display[:, :, :3]
                
                comparison = original_display.copy()
                mid_point = original_display.shape[1] // 2
                comparison[:, mid_point:, :] = enhanced_display[:, mid_point:, :]
                
                # Add a vertical line at the split point
                comparison[:, mid_point-1:mid_point+1, :] = 255
                
            else:
                logger.error(f"Unsupported comparison mode: {mode}")
                return None
            
            # Save the comparison image if output_path is provided
            if output_path:
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    Image.fromarray(comparison).save(output_path)
                    logger.info(f"Comparison image saved to {output_path}")
                    # Return the comparison instead of True
                except Exception as e:
                    logger.error(f"Error saving comparison image: {e}")
            
            return comparison
        
        except Exception as e:
            logger.error(f"Error creating comparison image: {e}")
            # Create a simple error image
            try:
                # Generate an "error" image with text if we can
                error_img = np.ones((300, 500, 3), dtype=np.uint8) * 220  # Light gray background
                error_img_pil = Image.fromarray(error_img)
                draw = ImageDraw.Draw(error_img_pil)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                        
                if font:
                    draw.text((20, 20), "Error Creating Comparison", fill=(255, 0, 0), font=font)
                    draw.text((20, 60), str(e), fill=(0, 0, 0), font=font)
                else:
                    # Draw some red pixels to indicate an error
                    error_img[10:30, 10:490] = [255, 0, 0]
                
                # If output_path provided, save this error image
                if output_path:
                    try:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        error_img_pil.save(output_path)
                    except:
                        pass
                
                return np.array(error_img_pil)
            except:
                # If even creating the error image fails, return a simple colored image
                return np.ones((300, 500, 3), dtype=np.uint8) * [255, 200, 200]  # Light red