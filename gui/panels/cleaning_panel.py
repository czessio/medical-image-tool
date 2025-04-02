"""
Cleaning panel component for medical image enhancement application.
Provides controls for image enhancement operations.
"""
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QCheckBox, QComboBox, QLabel, QSlider, QPushButton,
    QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)

class CleaningPanel(QWidget):
    """
    Control panel for image cleaning and enhancement options.
    
    Features:
    - Selection of different cleaning methods
    - Parameter adjustment for each method
    - Toggle for using novel vs foundational models
    """
    
    # Signals
    cleaningRequested = pyqtSignal(dict)  # Emitted when clean button is pressed
    
    def __init__(self, parent=None):
        """Initialize the cleaning panel."""
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        # Novel vs Foundational models switch
        self.use_novel_models = QCheckBox("Use Novel Models")
        self.use_novel_models.setChecked(True)
        self.use_novel_models.setToolTip("Switch between novel (cutting-edge) and foundational (established) models")
        model_layout.addWidget(self.use_novel_models)
        
        layout.addWidget(model_group)
        
        # Denoising group
        denoising_group = QGroupBox("Denoising")
        denoising_layout = QVBoxLayout(denoising_group)
        
        # Enable denoising checkbox
        self.enable_denoising = QCheckBox("Enable Denoising")
        self.enable_denoising.setChecked(True)
        denoising_layout.addWidget(self.enable_denoising)
        
        # Denoising strength slider
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Strength:"))
        
        self.denoising_strength = QSlider(Qt.Orientation.Horizontal)
        self.denoising_strength.setRange(1, 100)
        self.denoising_strength.setValue(50)
        strength_layout.addWidget(self.denoising_strength)
        
        self.denoising_strength_label = QLabel("50%")
        strength_layout.addWidget(self.denoising_strength_label)
        
        denoising_layout.addLayout(strength_layout)
        
        # Connect slider to label
        self.denoising_strength.valueChanged.connect(
            lambda v: self.denoising_strength_label.setText(f"{v}%")
        )
        
        layout.addWidget(denoising_group)
        
        # Super-resolution group
        sr_group = QGroupBox("Super-Resolution")
        sr_layout = QVBoxLayout(sr_group)
        
        # Enable super-resolution checkbox
        self.enable_sr = QCheckBox("Enable Super-Resolution")
        self.enable_sr.setChecked(True)
        sr_layout.addWidget(self.enable_sr)
        
        # Scale factor selection
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale Factor:"))
        
        self.sr_scale = QComboBox()
        self.sr_scale.addItem("1x (Enhance Only)", 1)
        self.sr_scale.addItem("2x", 2)
        self.sr_scale.addItem("3x", 3)
        self.sr_scale.addItem("4x", 4)
        self.sr_scale.setCurrentIndex(1)  # Default to 2x
        scale_layout.addWidget(self.sr_scale)
        
        sr_layout.addLayout(scale_layout)
        
        layout.addWidget(sr_group)
        
        # Artifact removal group
        artifact_group = QGroupBox("Artifact Removal")
        artifact_layout = QVBoxLayout(artifact_group)
        
        # Enable artifact removal checkbox
        self.enable_artifact = QCheckBox("Enable Artifact Removal")
        self.enable_artifact.setChecked(True)
        artifact_layout.addWidget(self.enable_artifact)
        
        # Artifact type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Target Artifacts:"))
        
        self.artifact_type = QComboBox()
        self.artifact_type.addItem("All Types", "all")
        self.artifact_type.addItem("Motion", "motion")
        self.artifact_type.addItem("Noise", "noise")
        self.artifact_type.addItem("Streaks", "streaks")
        type_layout.addWidget(self.artifact_type)
        
        artifact_layout.addLayout(type_layout)
        
        layout.addWidget(artifact_group)
        
        # Processing options group
        processing_group = QGroupBox("Processing Options")
        processing_layout = QVBoxLayout(processing_group)
        
        # Processing quality selection
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        
        self.processing_quality = QComboBox()
        self.processing_quality.addItem("Draft (Fast)", "draft")
        self.processing_quality.addItem("Standard", "standard")
        self.processing_quality.addItem("High (Slow)", "high")
        self.processing_quality.setCurrentIndex(1)  # Default to standard
        quality_layout.addWidget(self.processing_quality)
        
        processing_layout.addLayout(quality_layout)
        
        layout.addWidget(processing_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.clean_button = QPushButton("Clean Image")
        self.clean_button.clicked.connect(self._on_clean_clicked)
        button_layout.addWidget(self.clean_button)
        
        self.reset_button = QPushButton("Reset Options")
        self.reset_button.clicked.connect(self._reset_options)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch to bottom to keep controls at the top
        layout.addStretch()
    
    def _on_clean_clicked(self):
        """Handle clean button click."""
        # Collect all options
        options = {
            "use_novel_models": self.use_novel_models.isChecked(),
            
            "denoising": {
                "enabled": self.enable_denoising.isChecked(),
                "strength": self.denoising_strength.value() / 100.0
            },
            
            "super_resolution": {
                "enabled": self.enable_sr.isChecked(),
                "scale_factor": self.sr_scale.currentData()
            },
            
            "artifact_removal": {
                "enabled": self.enable_artifact.isChecked(),
                "artifact_type": self.artifact_type.currentData()
            },
            
            "processing": {
                "quality": self.processing_quality.currentData()
            }
        }
        
        # Emit signal with options
        logger.debug(f"Cleaning requested with options: {options}")
        self.cleaningRequested.emit(options)
    
    def _reset_options(self):
        """Reset all options to defaults."""
        self.use_novel_models.setChecked(True)
        
        self.enable_denoising.setChecked(True)
        self.denoising_strength.setValue(50)
        
        self.enable_sr.setChecked(True)
        self.sr_scale.setCurrentIndex(1)  # 2x
        
        self.enable_artifact.setChecked(True)
        self.artifact_type.setCurrentIndex(0)  # All types
        
        self.processing_quality.setCurrentIndex(1)  # Standard
    
def get_options(self):
    """
    Get the current cleaning options.
    
    Returns:
        dict: Dictionary of cleaning options
    """
    return {
        "use_novel_models": self.use_novel_models.isChecked(),
        
        "denoising": {
            "enabled": self.enable_denoising.isChecked(),
            "strength": self.denoising_strength.value() / 100.0
        },
        
        "super_resolution": {
            "enabled": self.enable_sr.isChecked(),
            "scale_factor": self.sr_scale.currentData()
        },
        
        "artifact_removal": {
            "enabled": self.enable_artifact.isChecked(),
            "artifact_type": self.artifact_type.currentData()
        },
        
        "processing": {
            "quality": self.processing_quality.currentData()
        }
    }