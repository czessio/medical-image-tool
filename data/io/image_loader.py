"""
General image loading functionality for the medical image enhancement application.
Handles loading, processing and saving of common image formats (PNG, JPEG, etc).
"""
import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image

from .dicom_handler import DicomHandler

logger = logging.getLogger(__name__)

class ImageLoader:
    """Handles loading of various image formats including medical images."""
    
    # Supported standard image formats
    STANDARD_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # Supported medical image formats
    MEDICAL_FORMATS = ['.dcm']
    
    @staticmethod
    def get_supported_formats():
        """Get a list of all supported file extensions."""
        return ImageLoader.STANDARD_FORMATS + ImageLoader.MEDICAL_FORMATS
    
    @staticmethod
    def is_supported_format(file_path):
        """Check if the file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in ImageLoader.get_supported_formats()
    
    @staticmethod
    def is_dicom(file_path):
        """Check if the file is a DICOM file based on extension."""
        ext = Path(file_path).suffix.lower()
        return ext in ImageLoader.MEDICAL_FORMATS
    
    @staticmethod
    def load_image(file_path):
        """
        Load an image file, automatically detecting the format.
    
        Args:
            file_path: Path to the image file
        
        Returns:
            tuple: (image_data, metadata, is_medical_format)
        """
        file_path = str(file_path)  # Convert Path objects to string
        logger.info(f"Loading image: {file_path}")
    
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
    
        # Check file extension after confirming file exists
        if not ImageLoader.is_supported_format(file_path):
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError(f"Unsupported file format: {file_path}")
    
        # Rest of the function remains the same...
        
        
        
        # Handle DICOM files
        if ImageLoader.is_dicom(file_path):
            try:
                image_data, metadata = DicomHandler.load_dicom(file_path)
                return image_data, metadata, True
            except ImportError:
                logger.error("DICOM handling is not available (pydicom not installed)")
                raise
            except Exception as e:
                logger.error(f"Error loading DICOM file: {e}")
                raise
        
        # Handle standard image formats using PIL
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary to ensure consistent channels
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                elif img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                # Convert to numpy array
                image_data = np.array(img)
                
                # Convert to float32 and normalize to 0-1
                image_data = image_data.astype(np.float32)
                if image_data.max() > 1.0:
                    image_data = image_data / 255.0
                
                # Extract basic metadata
                metadata = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'filename': os.path.basename(file_path),
                    'original_path': file_path
                }
                
                logger.info(f"Standard image loaded successfully: {image_data.shape}, {image_data.dtype}")
                return image_data, metadata, False
                
        except Exception as e:
            logger.error(f"Error loading standard image file: {e}")
            raise
    
    @staticmethod
    def save_image(image_data, output_path, metadata=None, is_medical_format=False):
        """
        Save image data to a file, choosing the appropriate format based on extension.
        
        Args:
            image_data: Numpy array containing the image data
            output_path: Path to save the image file
            metadata: Optional metadata dictionary
            is_medical_format: Whether to save as a medical format (DICOM)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            output_path = str(output_path)  # Convert Path objects to string
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Handle DICOM files
            if is_medical_format or Path(output_path).suffix.lower() == '.dcm':
                if not DicomHandler.is_available():
                    logger.error("DICOM handling is not available (pydicom not installed)")
                    return False
                
                return DicomHandler.save_dicom(image_data, metadata or {}, output_path)
            
            # Handle standard image formats
            # Ensure the image data is in the correct format for PIL
            if np.issubdtype(image_data.dtype, np.floating):
                # Convert from 0-1 float to 0-255 uint8
                image_data = (image_data * 255).clip(0, 255).astype(np.uint8)
            
            # Create PIL image
            img = Image.fromarray(image_data)
            
            # Save the image
            img.save(output_path)
            logger.info(f"Standard image saved successfully to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False