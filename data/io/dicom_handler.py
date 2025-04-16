"""
Enhancements to the DICOM handler to extract more metadata and improve display.
These updates extract and format DICOM metadata for better visualization.
"""

# Assuming there's an existing dicom_handler.py file, here's what we'll enhance

class DicomHandler:
    """Handler for DICOM image files with enhanced metadata extraction."""
    
    @staticmethod
    def load_dicom(file_path):
        """
        Load a DICOM image file and extract enhanced metadata.
        
        Args:
            file_path: Path to the DICOM file
            
        Returns:
            tuple: (image_data, metadata)
        """
        # Import pydicom conditionally to handle installations without it
        try:
            import pydicom
        except ImportError:
            raise ImportError("pydicom is required to load DICOM files")
        
        # Load the DICOM file
        dicom_data = pydicom.dcmread(file_path)
        
        # Extract the image data
        image_data = dicom_data.pixel_array
        
        # Apply scaling if available
        if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
            image_data = image_data * float(dicom_data.RescaleSlope) + float(dicom_data.RescaleIntercept)
        
        # Extract and format metadata
        metadata = DicomHandler.extract_metadata(dicom_data)
        
        # Store original path for reference
        metadata['original_path'] = file_path
        
        return image_data, metadata
    
    @staticmethod
    def extract_metadata(dicom_data):
        """
        Extract formatted metadata from a DICOM dataset.
        
        Args:
            dicom_data: pydicom Dataset object
            
        Returns:
            dict: Formatted metadata dictionary
        """
        metadata = {}
        
        # Helper function to safely get attributes
        def safe_get(dataset, attr, default=""):
            if hasattr(dataset, attr):
                value = getattr(dataset, attr)
                # Handle special cases
                if value == "":
                    return default
                return value
            return default
        
        # Extract key metadata fields
        # Patient information
        metadata['PatientName'] = str(safe_get(dicom_data, 'PatientName'))
        metadata['PatientID'] = safe_get(dicom_data, 'PatientID')
        metadata['PatientBirthDate'] = DicomHandler.format_date(safe_get(dicom_data, 'PatientBirthDate'))
        metadata['PatientSex'] = safe_get(dicom_data, 'PatientSex')
        metadata['PatientAge'] = safe_get(dicom_data, 'PatientAge')
        metadata['PatientWeight'] = safe_get(dicom_data, 'PatientWeight')
        metadata['PatientSize'] = safe_get(dicom_data, 'PatientSize')
        
        # Study information
        metadata['StudyDate'] = DicomHandler.format_date(safe_get(dicom_data, 'StudyDate'))
        metadata['StudyTime'] = DicomHandler.format_time(safe_get(dicom_data, 'StudyTime'))
        metadata['StudyDescription'] = safe_get(dicom_data, 'StudyDescription')
        metadata['StudyInstanceUID'] = safe_get(dicom_data, 'StudyInstanceUID')
        metadata['AccessionNumber'] = safe_get(dicom_data, 'AccessionNumber')
        metadata['ReferringPhysicianName'] = str(safe_get(dicom_data, 'ReferringPhysicianName'))
        metadata['StudyID'] = safe_get(dicom_data, 'StudyID')
        
        # Series information
        metadata['SeriesDate'] = DicomHandler.format_date(safe_get(dicom_data, 'SeriesDate'))
        metadata['SeriesTime'] = DicomHandler.format_time(safe_get(dicom_data, 'SeriesTime'))
        metadata['SeriesDescription'] = safe_get(dicom_data, 'SeriesDescription')
        metadata['SeriesNumber'] = safe_get(dicom_data, 'SeriesNumber')
        metadata['Modality'] = safe_get(dicom_data, 'Modality')
        metadata['BodyPartExamined'] = safe_get(dicom_data, 'BodyPartExamined')
        metadata['SeriesInstanceUID'] = safe_get(dicom_data, 'SeriesInstanceUID')
        
        # Image information
        metadata['InstanceNumber'] = safe_get(dicom_data, 'InstanceNumber')
        metadata['ImagePosition'] = str(safe_get(dicom_data, 'ImagePositionPatient', ''))
        metadata['ImageOrientation'] = str(safe_get(dicom_data, 'ImageOrientationPatient', ''))
        metadata['PixelSpacing'] = str(safe_get(dicom_data, 'PixelSpacing', ''))
        metadata['SliceThickness'] = safe_get(dicom_data, 'SliceThickness', '')
        metadata['SliceLocation'] = safe_get(dicom_data, 'SliceLocation', '')
        
        # Window settings
        metadata['WindowCenter'] = safe_get(dicom_data, 'WindowCenter', '')
        metadata['WindowWidth'] = safe_get(dicom_data, 'WindowWidth', '')
        
        # Rescale settings
        metadata['RescaleIntercept'] = safe_get(dicom_data, 'RescaleIntercept', '')
        metadata['RescaleSlope'] = safe_get(dicom_data, 'RescaleSlope', '')
        
        # Pixel data properties
        metadata['SamplesPerPixel'] = safe_get(dicom_data, 'SamplesPerPixel', '')
        metadata['PhotometricInterpretation'] = safe_get(dicom_data, 'PhotometricInterpretation', '')
        metadata['Rows'] = safe_get(dicom_data, 'Rows', '')
        metadata['Columns'] = safe_get(dicom_data, 'Columns', '')
        metadata['BitsAllocated'] = safe_get(dicom_data, 'BitsAllocated', '')
        metadata['BitsStored'] = safe_get(dicom_data, 'BitsStored', '')
        metadata['HighBit'] = safe_get(dicom_data, 'HighBit', '')
        metadata['PixelRepresentation'] = safe_get(dicom_data, 'PixelRepresentation', '')
        
        # Equipment information
        metadata['Manufacturer'] = safe_get(dicom_data, 'Manufacturer', '')
        metadata['ManufacturerModelName'] = safe_get(dicom_data, 'ManufacturerModelName', '')
        metadata['SoftwareVersions'] = safe_get(dicom_data, 'SoftwareVersions', '')
        metadata['StationName'] = safe_get(dicom_data, 'StationName', '')
        metadata['DeviceSerialNumber'] = safe_get(dicom_data, 'DeviceSerialNumber', '')
        metadata['InstitutionName'] = safe_get(dicom_data, 'InstitutionName', '')
        
        # Include all other attributes for completeness
        for elem in dicom_data:
            if elem.keyword and elem.keyword not in metadata:
                try:
                    # Skip pixel data (too large)
                    if elem.keyword == 'PixelData':
                        continue
                    
                    # Try to extract value
                    if elem.value != "":
                        metadata[elem.keyword] = str(elem.value)
                except Exception as e:
                    # Skip elements that can't be converted to string
                    pass
        
        return metadata
    
    @staticmethod
    def format_date(date_str):
        """
        Format DICOM date string (YYYYMMDD) to readable format (YYYY-MM-DD).
        
        Args:
            date_str: DICOM date string
            
        Returns:
            str: Formatted date string
        """
        if not date_str or len(date_str) != 8:
            return date_str
        
        try:
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            return f"{year}-{month}-{day}"
        except:
            return date_str
    
    @staticmethod
    def format_time(time_str):
        """
        Format DICOM time string (HHMMSS.FFFFFF) to readable format (HH:MM:SS).
        
        Args:
            time_str: DICOM time string
            
        Returns:
            str: Formatted time string
        """
        if not time_str:
            return time_str
        
        try:
            # Handle time strings with fractional seconds
            if '.' in time_str:
                time_str = time_str.split('.')[0]
            
            # Handle various lengths
            if len(time_str) < 6:
                time_str = time_str.ljust(6, '0')
            
            hour = time_str[:2]
            minute = time_str[2:4]
            second = time_str[4:6]
            return f"{hour}:{minute}:{second}"
        except:
            return time_str
    
    @staticmethod
    def apply_window_level(image, metadata, window=None, level=None):
        """
        Apply window/level (contrast/brightness) to the image.
        
        Args:
            image: Image array
            metadata: Image metadata
            window: Window width (contrast), if None use metadata
            level: Window center (brightness), if None use metadata
            
        Returns:
            numpy.ndarray: Windowed image
        """
        import numpy as np
        
        # Get window/level from metadata if not provided
        if window is None:
            window = metadata.get('WindowWidth', None)
            if window is not None:
                try:
                    # Handle multiple window values
                    if isinstance(window, list) or isinstance(window, tuple):
                        window = float(window[0])
                    else:
                        window = float(window)
                except:
                    window = None
        
        if level is None:
            level = metadata.get('WindowCenter', None)
            if level is not None:
                try:
                    # Handle multiple level values
                    if isinstance(level, list) or isinstance(level, tuple):
                        level = float(level[0])
                    else:
                        level = float(level)
                except:
                    level = None
        
        # If window/level not available, return original image
        if window is None or level is None:
            return image
        
        # Apply window/level
        low = level - window/2
        high = level + window/2
        
        # Clip image to window range
        windowed_image = np.clip(image, low, high)
        
        # Normalize to [0, 1]
        if high != low:
            windowed_image = (windowed_image - low) / (high - low)
        else:
            windowed_image = np.zeros_like(image)
        
        return windowed_image