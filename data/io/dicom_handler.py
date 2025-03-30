"""
DICOM file handling for the medical image enhancement application.
Provides functionality to load, process, and save DICOM images.
"""
import os
import logging
from pathlib import Path
import numpy as np




try:
    import pydicom
    # Try a different approach instead of importing apply_window which doesn't exist
    # from pydicom.pixel_data_handlers.util import apply_window
    PYDICOM_AVAILABLE = True
    print(f"DEBUG: pydicom successfully imported: {pydicom.__version__}")
except ImportError as e:
    PYDICOM_AVAILABLE = False
    print(f"DEBUG: pydicom import failed: {e}")




logger = logging.getLogger(__name__)

class DicomHandler:
    """Handles loading, processing and saving of DICOM files."""
    
    @staticmethod
    def is_available():
        """Check if DICOM handling is available (pydicom installed)."""
        return PYDICOM_AVAILABLE
    
    @staticmethod
    def check_availability():
        """Verify DICOM handling availability and raise error if not available."""
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM handling but is not installed.")
    
    @staticmethod
    def load_dicom(path):
        """
        Load a DICOM file and convert it to a normalized numpy array.
        
        Args:
            path: Path to DICOM file
            
        Returns:
            tuple: (image_data, metadata)
                - image_data: Numpy array containing the pixel data
                - metadata: Dictionary containing relevant DICOM metadata
        """
        DicomHandler.check_availability()
        
        logger.info(f"Loading DICOM file: {path}")
        try:
            # Load the DICOM file
            dicom_data = pydicom.dcmread(path)
            
            # Extract pixel data as numpy array
            image_data = dicom_data.pixel_array
            
            # Convert to float32 for processing
            image_data = image_data.astype(np.float32)
            
            # Apply rescaling if available
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                image_data = image_data * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
            
            # Extract relevant metadata
            metadata = DicomHandler.extract_metadata(dicom_data)
            
            logger.info(f"DICOM loaded successfully: {image_data.shape}, {image_data.dtype}")
            return image_data, metadata
            
        except Exception as e:
            logger.error(f"Error loading DICOM file: {e}")
            raise
    
    @staticmethod
    def apply_window_level(image_data, metadata, window=None, level=None):
        """
        Apply windowing (contrast adjustment) to the DICOM image.
        
        Args:
            image_data: The image data array
            metadata: DICOM metadata containing window/level information
            window: Window width (if None, use value from metadata if available)
            level: Window center (if None, use value from metadata if available)
            
        Returns:
            numpy.ndarray: Windowed image data
        """
        DicomHandler.check_availability()
        
        # Use provided window/level or get from metadata
        if window is None and 'WindowWidth' in metadata:
            window = metadata['WindowWidth']
        if level is None and 'WindowCenter' in metadata:
            level = metadata['WindowCenter']
            
        # If still None, use image min/max
        if window is None or level is None:
            min_val = np.min(image_data)
            max_val = np.max(image_data)
            if window is None:
                window = max_val - min_val
            if level is None:
                level = min_val + window / 2
        
        logger.debug(f"Applying window/level: {window}/{level}")
        
        # Apply windowing
        low = level - window / 2
        high = level + window / 2
        windowed_image = np.clip(image_data, low, high)
        
        # Normalize to 0-1 range
        windowed_image = (windowed_image - low) / (high - low)
        
        return windowed_image
    
    @staticmethod
    def save_dicom(image_data, metadata, output_path, original_dicom=None):
        """
        Save processed image data as a DICOM file.
        
        Args:
            image_data: Processed image data as numpy array
            metadata: Dictionary of DICOM metadata
            output_path: Path to save the new DICOM file
            original_dicom: Optional original DICOM object to use as template
            
        Returns:
            bool: True if successful, False otherwise
        """
        DicomHandler.check_availability()
        
        try:
            if original_dicom is None and 'original_dicom_path' in metadata:
                # Load the original DICOM as template
                original_dicom = pydicom.dcmread(metadata['original_dicom_path'])
            
            if original_dicom:
                # Use original as template
                ds = original_dicom
            else:
                # Create a new DICOM dataset
                ds = pydicom.Dataset()
                ds.file_meta = pydicom.Dataset()
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                
                # Set minimal required DICOM attributes
                ds.SOPClassUID = pydicom.uid.CT_Image_Storage
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                ds.StudyInstanceUID = pydicom.uid.generate_uid()
                ds.SeriesInstanceUID = pydicom.uid.generate_uid()
                
                # Add metadata from the provided dictionary
                for key, value in metadata.items():
                    if key not in ['original_dicom_path']:  # Skip non-DICOM metadata
                        try:
                            setattr(ds, key, value)
                        except:
                            logger.warning(f"Could not set DICOM attribute: {key}")
            
            # Ensure the image data is in the correct format
            # Convert from float to appropriate integer type
            if np.issubdtype(image_data.dtype, np.floating):
                # Scale to appropriate bit depth
                bit_depth = 16  # Default to 16-bit
                if 'BitsStored' in metadata:
                    bit_depth = metadata['BitsStored']
                
                max_val = 2**bit_depth - 1
                image_data = np.clip(image_data * max_val, 0, max_val).astype(np.uint16)
            
            # Update the pixel data
            ds.PixelData = image_data.tobytes()
            ds.Rows, ds.Columns = image_data.shape[:2]
            
            # Add processing information
            ds.ProcessingDescription = "Enhanced with Medical Image Enhancement Tool"
            
            # Save the file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ds.save_as(output_path)
            
            logger.info(f"DICOM saved successfully to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving DICOM file: {e}")
            return False
    
    @staticmethod
    def extract_metadata(dicom_data):
        """
        Extract relevant metadata from a DICOM dataset.
        
        Args:
            dicom_data: pydicom.dataset.FileDataset object
            
        Returns:
            dict: Dictionary of relevant DICOM metadata
        """
        metadata = {
            'original_dicom_path': getattr(dicom_data, 'filename', None),
            'PatientID': getattr(dicom_data, 'PatientID', ''),
            'PatientName': str(getattr(dicom_data, 'PatientName', '')),
            'StudyDescription': getattr(dicom_data, 'StudyDescription', ''),
            'SeriesDescription': getattr(dicom_data, 'SeriesDescription', ''),
            'Modality': getattr(dicom_data, 'Modality', ''),
            'PixelSpacing': getattr(dicom_data, 'PixelSpacing', [1.0, 1.0]),
        }
        
        # Add window/level if available
        if hasattr(dicom_data, 'WindowWidth'):
            if isinstance(dicom_data.WindowWidth, list):
                metadata['WindowWidth'] = dicom_data.WindowWidth[0]
            else:
                metadata['WindowWidth'] = dicom_data.WindowWidth
                
        if hasattr(dicom_data, 'WindowCenter'):
            if isinstance(dicom_data.WindowCenter, list):
                metadata['WindowCenter'] = dicom_data.WindowCenter[0]
            else:
                metadata['WindowCenter'] = dicom_data.WindowCenter
        
        # Add bit depth information
        if hasattr(dicom_data, 'BitsStored'):
            metadata['BitsStored'] = dicom_data.BitsStored
        
        return metadata