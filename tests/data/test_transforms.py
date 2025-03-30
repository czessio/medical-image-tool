import sys
from pathlib import Path
import numpy as np
import pytest

# Add the parent directory to sys.path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.processing.transforms import (
    resize_image, normalize_image, adjust_window_level, 
    ensure_channel_first, ensure_channel_last
)

class TestTransforms:
    def setup_method(self):
        """Create test images before each test."""
        # Create a test RGB image
        self.rgb_image = np.zeros((100, 150, 3), dtype=np.float32)
        self.rgb_image[30:70, 50:100, 0] = 0.8  # Red rectangle
        self.rgb_image[20:80, 20:130, 1] = 0.5  # Green rectangle
        
        # Create a test grayscale image
        self.gray_image = np.zeros((100, 150), dtype=np.float32)
        self.gray_image[30:70, 50:100] = 0.8  # White rectangle
    
    def test_resize_image(self):
        """Test image resizing."""
        # Test with preserving aspect ratio
        resized = resize_image(self.rgb_image, (75, 50), preserve_aspect_ratio=True)
        assert resized.shape == (50, 75, 3)
        
        # Test without preserving aspect ratio
        resized = resize_image(self.rgb_image, (75, 50), preserve_aspect_ratio=False)
        assert resized.shape == (50, 75, 3)
        
        # Test with grayscale image
        resized = resize_image(self.gray_image, (75, 50))
        # Most resize implementations will add a channel dimension for PIL/RGB conversion
        assert resized.shape[:2] == (50, 75)
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Create an image with known range
        test_image = np.linspace(10, 50, 100).reshape(10, 10)
        
        # Normalize to [0, 1]
        normalized = normalize_image(test_image)
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
        
        # Normalize to custom range
        normalized = normalize_image(test_image, target_min=-1.0, target_max=2.0)
        assert normalized.min() == pytest.approx(-1.0)
        assert normalized.max() == pytest.approx(2.0)
        
        # Test with constant image
        constant_image = np.ones((10, 10)) * 5
        normalized = normalize_image(constant_image)
        assert np.all(normalized == 5)
    
    def test_adjust_window_level(self):
        """Test window/level adjustment."""
        # Create a test image with values from 0 to 100
        test_image = np.linspace(0, 100, 100).reshape(10, 10)
        
        # Apply windowing
        windowed = adjust_window_level(test_image, window=50, level=50)
        
        # Should clip values to range [25, 75] and normalize to [0, 1]
        assert windowed.min() == pytest.approx(0.0)
        assert windowed.max() == pytest.approx(1.0)
        
        # The original value 25 should map to 0
        assert windowed[2, 5] == pytest.approx(0.0)
        
        # The original value 75 should map to 1
        assert windowed[7, 5] == pytest.approx(1.0)
    
    def test_ensure_channel_first(self):
        """Test channel first conversion."""
        # Test with RGB image (H,W,C) -> (C,H,W)
        channel_first = ensure_channel_first(self.rgb_image)
        assert channel_first.shape == (3, 100, 150)
        
        # Test with grayscale image (H,W) -> (1,H,W)
        channel_first = ensure_channel_first(self.gray_image)
        assert channel_first.shape == (1, 100, 150)
    
    def test_ensure_channel_last(self):
        """Test channel last conversion."""
        # Create a channel-first RGB image
        channel_first_rgb = np.zeros((3, 100, 150), dtype=np.float32)
        
        # Convert to channel-last
        channel_last = ensure_channel_last(channel_first_rgb)
        assert channel_last.shape == (100, 150, 3)
        
        # Test with grayscale image (H,W) -> (H,W,1)
        channel_last = ensure_channel_last(self.gray_image)
        assert channel_last.shape == (100, 150, 1)