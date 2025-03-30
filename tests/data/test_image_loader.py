import os
import tempfile
from pathlib import Path
import sys
import numpy as np
import pytest
from PIL import Image

# Add the parent directory to sys.path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.io.image_loader import ImageLoader

class TestImageLoader:
    def setup_method(self):
        """Create test images before each test."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create a standard test image (RGB)
        self.rgb_path = os.path.join(self.test_dir, "test_rgb.png")
        rgb_array = np.zeros((100, 150, 3), dtype=np.uint8)
        rgb_array[30:70, 50:100, 0] = 255  # Red rectangle
        rgb_array[20:80, 20:130, 1] = 128  # Green rectangle
        Image.fromarray(rgb_array).save(self.rgb_path)
        
        # Create a grayscale test image
        self.gray_path = os.path.join(self.test_dir, "test_gray.png")
        gray_array = np.zeros((100, 150), dtype=np.uint8)
        gray_array[30:70, 50:100] = 200  # White rectangle
        Image.fromarray(gray_array).save(self.gray_path)
    
    def teardown_method(self):
        """Clean up temporary files after each test."""
        for file in Path(self.test_dir).glob("*"):
            try:
                os.remove(file)
            except:
                pass
        os.rmdir(self.test_dir)
    
    def test_supported_formats(self):
        """Test that the supported formats list is not empty."""
        formats = ImageLoader.get_supported_formats()
        assert len(formats) > 0
        assert '.png' in formats
        assert '.jpg' in formats
    
    def test_is_supported_format(self):
        """Test format detection."""
        assert ImageLoader.is_supported_format("test.png") == True
        assert ImageLoader.is_supported_format("test.jpg") == True
        assert ImageLoader.is_supported_format("test.dcm") == True
        assert ImageLoader.is_supported_format("test.doc") == False
    
    def test_is_dicom(self):
        """Test DICOM format detection."""
        assert ImageLoader.is_dicom("test.dcm") == True
        assert ImageLoader.is_dicom("test.png") == False
    
    def test_load_rgb_image(self):
        """Test loading an RGB image."""
        image_data, metadata, is_medical = ImageLoader.load_image(self.rgb_path)
        
        # Check image data
        assert isinstance(image_data, np.ndarray)
        assert image_data.shape == (100, 150, 3)
        assert image_data.dtype == np.float32
        assert 0 <= image_data.min() <= image_data.max() <= 1.0
        
        # Check metadata
        assert metadata['format'] == 'PNG'
        assert metadata['filename'] == 'test_rgb.png'
        
        # Check format flag
        assert is_medical == False
    
    def test_load_grayscale_image(self):
        """Test loading a grayscale image."""
        image_data, metadata, is_medical = ImageLoader.load_image(self.gray_path)
        
        # Check image data
        assert isinstance(image_data, np.ndarray)
        # Most loaders will convert grayscale to RGB, but some might not
        assert image_data.shape[:2] == (100, 150)  
        assert image_data.dtype == np.float32
        assert 0 <= image_data.min() <= image_data.max() <= 1.0
        
        # Check metadata
        assert metadata['format'] == 'PNG'
        assert metadata['filename'] == 'test_gray.png'
        
        # Check format flag
        assert is_medical == False
    
    def test_save_image(self):
        """Test saving an image."""
        # Load an image
        image_data, metadata, _ = ImageLoader.load_image(self.rgb_path)
        
        # Save it to a new path
        output_path = os.path.join(self.test_dir, "output.png")
        result = ImageLoader.save_image(image_data, output_path)
        
        # Check result
        assert result == True
        assert os.path.exists(output_path)
        
        # Load saved image to verify
        saved_image = np.array(Image.open(output_path))
        assert saved_image.shape[:2] == image_data.shape[:2]
    
    def test_error_handling(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError):
            ImageLoader.load_image("non_existent_file.png")
    
        # Create a file with invalid extension
        invalid_path = os.path.join(self.test_dir, "invalid_extension.xyz")
        with open(invalid_path, 'w') as f:
            f.write("test")
    
        with pytest.raises(ValueError):
            ImageLoader.load_image(invalid_path)