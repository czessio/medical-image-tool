import os
import tempfile
from pathlib import Path
import sys
import numpy as np
import pytest
from PIL import Image

# Add the parent directory to sys.path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.io.export import Exporter

class TestExporter:
    def setup_method(self):
        """Create test images before each test."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create an original test image
        self.original = np.zeros((100, 150, 3), dtype=np.float32)
        self.original[30:70, 50:100, 0] = 0.8  # Red rectangle
        
        # Create an enhanced test image
        self.enhanced = np.zeros((100, 150, 3), dtype=np.float32)
        self.enhanced[30:70, 50:100, 2] = 0.8  # Blue rectangle
    
    def teardown_method(self):
        """Clean up temporary files after each test."""
        for file in Path(self.test_dir).glob("*"):
            try:
                os.remove(file)
            except:
                pass
        os.rmdir(self.test_dir)
    
    def test_export_image(self):
        """Test exporting an image."""
        output_path = os.path.join(self.test_dir, "exported.png")
        
        # Export the enhanced image
        result = Exporter.export_image(self.enhanced, output_path)
        
        # Check result
        assert result == True
        assert os.path.exists(output_path)
        
        # Verify the exported image
        exported = np.array(Image.open(output_path))
        assert exported.shape[:2] == self.enhanced.shape[:2]
    
    def test_create_comparison_side_by_side(self):
        """Test side-by-side comparison image creation."""
        output_path = os.path.join(self.test_dir, "comparison_side_by_side.png")
        
        # Create a side-by-side comparison
        comparison = Exporter.create_comparison_image(
            self.original, self.enhanced, output_path, mode='side_by_side'
        )
        
        # Check the comparison image
        assert comparison.shape[0] == self.original.shape[0]  # Same height
        assert comparison.shape[1] > self.original.shape[1]   # Greater width
        assert os.path.exists(output_path)
    
    def test_create_comparison_overlay(self):
        """Test overlay comparison image creation."""
        output_path = os.path.join(self.test_dir, "comparison_overlay.png")
        
        # Create an overlay comparison
        comparison = Exporter.create_comparison_image(
            self.original, self.enhanced, output_path, mode='overlay'
        )
        
        # Check the comparison image
        assert comparison.shape == self.original.shape
        assert os.path.exists(output_path)
    
    def test_create_comparison_split(self):
        """Test split comparison image creation."""
        output_path = os.path.join(self.test_dir, "comparison_split.png")
        
        # Create a split comparison
        comparison = Exporter.create_comparison_image(
            self.original, self.enhanced, output_path, mode='split'
        )
        
        # Check the comparison image
        assert comparison.shape == self.original.shape
        assert os.path.exists(output_path)
    
    def test_comparison_invalid_mode(self):
        """Test error handling for invalid comparison mode."""
        # Try with an invalid mode
        comparison = Exporter.create_comparison_image(
            self.original, self.enhanced, mode='invalid_mode'
        )
        
        # Should return None for invalid mode
        assert comparison is None