import os
import tempfile
from pathlib import Path
import sys
import numpy as np
import pytest

# Add the parent directory to sys.path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.io.dicom_handler import DicomHandler

# Skip all tests if pydicom is not available
pytestmark = pytest.mark.skipif(
    not DicomHandler.is_available(),
    reason="pydicom is not installed"
)

class TestDicomHandler:
    def setup_method(self):
        """Set up test environment."""
        # This test requires pydicom to be installed
        if not DicomHandler.is_available():
            pytest.skip("pydicom is not installed")
        
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary files after each test."""
        for file in Path(self.test_dir).glob("*"):
            try:
                os.remove(file)
            except:
                pass
        os.rmdir(self.test_dir)
    
    def test_extract_metadata(self):
        """Test metadata extraction from DICOM dataset."""
        # This test requires a real DICOM file or mock
        # Since we don't have a real DICOM file for testing, we'll just
        # test that the function runs without error when proper mocks are used
        
        # Mock a minimal DICOM dataset
        import pydicom
        ds = pydicom.Dataset()
        ds.PatientID = "TEST123"
        ds.PatientName = "Test^Patient"
        ds.Modality = "CT"
        ds.WindowWidth = 400
        ds.WindowCenter = 40
        
        # Extract metadata
        metadata = DicomHandler.extract_metadata(ds)
        
        # Check key fields
        assert metadata['PatientID'] == "TEST123"
        assert metadata['PatientName'] == "Test^Patient"
        assert metadata['Modality'] == "CT"
        assert metadata['WindowWidth'] == 400
        assert metadata['WindowCenter'] == 40
    
    def test_apply_window_level(self):
        """Test window/level adjustment."""
        # Create a test image data array
        image_data = np.linspace(0, 1000, 100*100).reshape(100, 100)
        
        # Create metadata with window/level
        metadata = {
            'WindowWidth': 500,
            'WindowCenter': 500
        }
        
        # Apply windowing
        windowed = DicomHandler.apply_window_level(image_data, metadata)
        
        # Check the resulting image
        assert windowed.min() == 0.0   # Values below window should be 0
        assert windowed.max() == 1.0   # Values above window should be 1
        
        # Test with explicit window/level
        windowed = DicomHandler.apply_window_level(image_data, {}, window=200, level=800)
        
        # Check the resulting image
        assert windowed.min() == 0.0
        assert windowed.max() == 1.0