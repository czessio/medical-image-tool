"""
DICOM metadata viewer component for medical image enhancement application.
Displays DICOM metadata in a structured format for medical professionals.
"""
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTableWidget, QTableWidgetItem, QPushButton,
    QGroupBox, QComboBox, QLineEdit, QCheckBox,
    QTabWidget, QSplitter, QFrame, QHeaderView,
    QDialog, QDialogButtonBox, QScrollArea
)
from PyQt6.QtGui import QFont, QIcon, QColor
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize

logger = logging.getLogger(__name__)

class DicomMetadataViewer(QWidget):
    """
    Widget for displaying DICOM metadata.
    
    Features:
    - Tabbed display of different metadata categories
    - Search and filter capabilities
    - Highlighted display of key parameters
    - Copy to clipboard functionality
    """
    
    def __init__(self, parent=None):
        """Initialize the DICOM metadata viewer."""
        super().__init__(parent)
        self.metadata = None
        self.current_filter = ""
        
        # Define categories for organizing DICOM tags
        self.categories = {
            "Patient": [
                "PatientName", "PatientID", "PatientBirthDate", "PatientSex", 
                "PatientAge", "PatientWeight", "PatientSize"
            ],
            "Study": [
                "StudyDate", "StudyTime", "StudyDescription", "StudyInstanceUID",
                "AccessionNumber", "ReferringPhysicianName", "StudyID"
            ],
            "Series": [
                "SeriesDate", "SeriesTime", "SeriesDescription", "SeriesNumber",
                "Modality", "BodyPartExamined", "SeriesInstanceUID"
            ],
            "Image": [
                "InstanceNumber", "ImagePosition", "ImageOrientation", "PixelSpacing",
                "SliceThickness", "SliceLocation", "WindowCenter", "WindowWidth",
                "RescaleIntercept", "RescaleSlope", "SamplesPerPixel", "PhotometricInterpretation"
            ],
            "Equipment": [
                "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
                "StationName", "DeviceSerialNumber", "InstitutionName"
            ]
        }
        
        # Key parameters to highlight
        self.key_parameters = [
            "Modality", "PatientName", "PatientID", "StudyDate", "SeriesDescription",
            "WindowCenter", "WindowWidth", "RescaleIntercept", "RescaleSlope"
        ]
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title and controls
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("DICOM Metadata")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        # Filter input
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        filter_layout.addWidget(filter_label)
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter tags...")
        self.filter_input.textChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_input)
        
        header_layout.addLayout(filter_layout)
        
        # Controls for view options
        view_layout = QHBoxLayout()
        
        # Create tab widget for categories
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)  # Makes tabs look cleaner
        
        # Create tabs for each category
        self.category_tables = {}
        
        for category, tags in self.categories.items():
            # Create table for this category
            table = QTableWidget(0, 2)  # 0 rows initially, 2 columns (Tag, Value)
            table.setHorizontalHeaderLabels(["Tag", "Value"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            
            # Add to tab widget
            self.tab_widget.addTab(table, category)
            
            # Store for later access
            self.category_tables[category] = table
        
        # Add "All" tab for all metadata
        all_table = QTableWidget(0, 2)
        all_table.setHorizontalHeaderLabels(["Tag", "Value"])
        all_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        all_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        all_table.setAlternatingRowColors(True)
        all_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        all_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tab_widget.addTab(all_table, "All")
        self.category_tables["All"] = all_table
        
        # Add to layout
        layout.addLayout(header_layout)
        layout.addWidget(self.tab_widget)
        
        # Status bar
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("No DICOM metadata loaded")
        status_layout.addWidget(self.status_label)
        
        # Add copy button
        self.copy_button = QPushButton("Copy Selected")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self._on_copy_selected)
        status_layout.addWidget(self.copy_button)
        
        # Add export button
        self.export_button = QPushButton("Export Metadata")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export_clicked)
        status_layout.addWidget(self.export_button)
        
        layout.addWidget(status_frame)
        
        # Set tab widget as the focus
        self.tab_widget.setFocus()
    
    def set_metadata(self, metadata):
        """
        Set the DICOM metadata to display.
        
        Args:
            metadata: Dictionary of DICOM metadata
        """
        self.metadata = metadata
        
        # Clear current tables
        for table in self.category_tables.values():
            table.setRowCount(0)
        
        # If no metadata, disable controls
        if not metadata:
            self.status_label.setText("No DICOM metadata loaded")
            self.copy_button.setEnabled(False)
            self.export_button.setEnabled(False)
            return
        
        # Enable controls
        self.copy_button.setEnabled(True)
        self.export_button.setEnabled(True)
        
        # Update status
        modality = metadata.get('Modality', 'Unknown')
        patient_id = metadata.get('PatientID', 'Unknown')
        study_date = metadata.get('StudyDate', 'Unknown')
        
        self.status_label.setText(f"Modality: {modality} | Patient: {patient_id} | Date: {study_date}")
        
        # Populate tables for each category
        for category, tags in self.categories.items():
            table = self.category_tables[category]
            
            # Add rows for tags in this category
            row = 0
            for tag in tags:
                if tag in metadata and self._matches_filter(tag, metadata[tag]):
                    table.insertRow(row)
                    
                    # Tag name
                    tag_item = QTableWidgetItem(tag)
                    
                    # Highlight key parameters
                    if tag in self.key_parameters:
                        tag_item.setBackground(QColor(240, 248, 255))  # Light blue background
                        tag_item.setForeground(QColor(0, 0, 128))      # Dark blue text
                        font = tag_item.font()
                        font.setBold(True)
                        tag_item.setFont(font)
                    
                    table.setItem(row, 0, tag_item)
                    
                    # Tag value
                    value_item = QTableWidgetItem(str(metadata[tag]))
                    table.setItem(row, 1, value_item)
                    
                    row += 1
        
        # Populate the "All" tab with all metadata
        all_table = self.category_tables["All"]
        row = 0
        
        for tag, value in sorted(metadata.items()):
            if self._matches_filter(tag, value):
                all_table.insertRow(row)
                
                # Tag name
                tag_item = QTableWidgetItem(tag)
                
                # Highlight key parameters
                if tag in self.key_parameters:
                    tag_item.setBackground(QColor(240, 248, 255))  # Light blue background
                    tag_item.setForeground(QColor(0, 0, 128))      # Dark blue text
                    font = tag_item.font()
                    font.setBold(True)
                    tag_item.setFont(font)
                
                all_table.setItem(row, 0, tag_item)
                
                # Tag value
                value_item = QTableWidgetItem(str(value))
                all_table.setItem(row, 1, value_item)
                
                row += 1
    
    def _matches_filter(self, tag, value):
        """
        Check if a tag and value match the current filter.
        
        Args:
            tag: Tag name
            value: Tag value
            
        Returns:
            bool: True if matches filter, False otherwise
        """
        if not self.current_filter:
            return True
            
        filter_lower = self.current_filter.lower()
        return (filter_lower in tag.lower() or 
                filter_lower in str(value).lower())
    
    def _on_filter_changed(self, text):
        """
        Handle filter text change.
        
        Args:
            text: New filter text
        """
        self.current_filter = text
        
        # Reapply metadata with the new filter
        if self.metadata:
            self.set_metadata(self.metadata)
    
    def _on_copy_selected(self):
        """Handle copy button click to copy selected metadata to clipboard."""
        # Get the current tab and table
        current_tab = self.tab_widget.currentWidget()
        
        if not current_tab or not isinstance(current_tab, QTableWidget):
            return
            
        # Get selected rows
        selected_ranges = current_tab.selectedRanges()
        if not selected_ranges:
            return
            
        # Prepare text for clipboard
        clipboard_text = []
        
        for range_obj in selected_ranges:
            for row in range(range_obj.topRow(), range_obj.bottomRow() + 1):
                tag_item = current_tab.item(row, 0)
                value_item = current_tab.item(row, 1)
                
                if tag_item and value_item:
                    clipboard_text.append(f"{tag_item.text()}: {value_item.text()}")
        
        # Copy to clipboard
        if clipboard_text:
            from PyQt6.QtGui import QGuiApplication
            QGuiApplication.clipboard().setText('\n'.join(clipboard_text))
            self.status_label.setText(f"Copied {len(clipboard_text)} tags to clipboard")
    
    def _on_export_clicked(self):
        """Handle export button click to export metadata to file."""
        if not self.metadata:
            return
            
        # Show file dialog
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export DICOM Metadata", "", 
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Determine file type from extension
            if file_path.lower().endswith('.csv'):
                # CSV format
                with open(file_path, 'w') as f:
                    f.write("Tag,Value\n")
                    for tag, value in sorted(self.metadata.items()):
                        # Handle commas in values by quoting
                        value_str = f"\"{value}\"" if ',' in str(value) else str(value)
                        f.write(f"{tag},{value_str}\n")
            else:
                # Text format
                with open(file_path, 'w') as f:
                    # Add a header
                    f.write("DICOM Metadata Export\n")
                    f.write("=====================\n\n")
                    
                    # Group by categories
                    for category, tags in self.categories.items():
                        f.write(f"{category}:\n")
                        f.write("-" * len(category) + "\n")
                        
                        for tag in tags:
                            if tag in self.metadata:
                                f.write(f"{tag}: {self.metadata[tag]}\n")
                        
                        f.write("\n")
                    
                    # Write remaining tags
                    f.write("Other Tags:\n")
                    f.write("-----------\n")
                    
                    category_tags = [tag for tags in self.categories.values() for tag in tags]
                    other_tags = {tag: value for tag, value in self.metadata.items() 
                                if tag not in category_tags}
                    
                    for tag, value in sorted(other_tags.items()):
                        f.write(f"{tag}: {value}\n")
            
            self.status_label.setText(f"Metadata exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metadata: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export Error", f"Error exporting metadata: {str(e)}")
    
    def sizeHint(self):
        """Suggested size for the widget."""
        return QSize(600, 400)