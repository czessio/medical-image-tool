import sys
from pathlib import Path
import numpy as np
import pytest

# Add the parent directory to sys.path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.processing.visualization import (
    create_thumbnail, draw_info_overlay, create_histogram, overlay_mask
)

class TestVisualization:
    def setup_method(self):
        """Create test images before each test."""
        # Create a test RGB image
        self.rgb_image = np.zeros((100, 150, 3), dtype=np.float32)
        self.rgb_image[30:70, 50:100, 0] = 0.8  # Red rectangle
        self.rgb_image[20:80, 20:130, 1] = 0.5  # Green rectangle
        
        # Create a test mask
        self.mask = np.zeros((100, 150), dtype=bool)
        self.mask[40:60, 60:90] = True  # Small rectangle in the center
    
    def test_create_thumbnail(self):
        """Test thumbnail creation."""
        # Create a thumbnail
        thumbnail = create_thumbnail(self.rgb_image, size=(75, 50))
        
        # Check thumbnail size (should be smaller but preserve aspect ratio)
        assert thumbnail.shape[0] <= 50
        assert thumbnail.shape[1] <= 75
        
        # Test with uint8 image
        uint8_image = (self.rgb_image * 255).astype(np.uint8)
        thumbnail = create_thumbnail(uint8_image, size=(75, 50))
        assert thumbnail.shape[0] <= 50
        assert thumbnail.shape[1] <= 75
    
    def test_draw_info_overlay(self):
        """Test drawing information overlay."""
        # Draw overlay with text
        overlay = draw_info_overlay(self.rgb_image, "Test Overlay")
        
        # Should return an image of the same size
        assert overlay.shape[:2] == self.rgb_image.shape[:2]
    
    def test_create_histogram(self):
        """Test histogram creation."""
        # Create histogram for the entire image
        hist, bins = create_histogram(self.rgb_image)
        
        # Check histogram shape
        assert len(hist) == 255  # Default number of bins - 1
        assert len(bins) == 256  # Number of bin edges
        
        # Test with specific channel
        hist, bins = create_histogram(self.rgb_image, channel=0)
        assert len(hist) == 255
        
        # Test with grayscale image
        gray_image = self.rgb_image.mean(axis=2)
        hist, bins = create_histogram(gray_image)
        assert len(hist) == 255
    
    def test_overlay_mask(self):
        """Test mask overlay."""
        # Apply mask overlay
        result = overlay_mask(self.rgb_image, self.mask)
        
        # Result should have same shape as input (except potentially alpha channel)
        assert result.shape[:2] == self.rgb_image.shape[:2]
        
        # Check that the masked region has a different color than the original
        masked_region = result[self.mask]
        original_region = self.rgb_image[self.mask]
        assert not np.array_equal(masked_region, original_region)