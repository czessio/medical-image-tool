"""
Cleaning panel component for medical image enhancement application.
Provides controls for image enhancement operations with improved model selection.
"""
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QCheckBox, QComboBox, QLabel, QSlider, QPushButton,
    QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
    QFrame, QTabWidget, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon, QFont

from utils.config import Config

logger = logging.getLogger(__name__)

class CleaningPanel(QWidget):
    """
    Control panel for image cleaning and enhancement options.
    
    Features:
    - Clear selection of different model types (foundational vs. novel)
    - Adjustable parameters for each cleaning method
    - Visual feedback for active models
    """
    
    # Signals
    cleaningRequested = pyqtSignal(dict)  # Emitted when clean button is pressed
    
    def __init__(self, parent=None):
        """Initialize the cleaning panel."""
        super().__init__(parent)
        self.config = Config()
        self._init_ui()
        self._load_config()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        
        # Title label
        title_label = QLabel("Image Enhancement Settings")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.models_tab = QWidget()
        self.settings_tab = QWidget()
        
        self.tab_widget.addTab(self.models_tab, "Models")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        
        # Initialize tab contents
        self._init_models_tab()
        self._init_settings_tab()
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.clean_button = QPushButton("Enhance Image")
        self.clean_button.setMinimumHeight(40)
        self.clean_button.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.clean_button.clicked.connect(self._on_clean_clicked)
        button_layout.addWidget(self.clean_button)
        
        self.reset_button = QPushButton("Reset Options")
        self.reset_button.setProperty("secondary", "true")  # For stylesheet
        self.reset_button.clicked.connect(self._reset_options)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch to bottom to keep controls at the top
        layout.addStretch()
    
    def _init_models_tab(self):
        """Initialize the models tab content."""
        layout = QVBoxLayout(self.models_tab)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(15)
        
        # Model selection group
        model_group = QGroupBox("Model Type")
        model_layout = QVBoxLayout(model_group)
        
        # Model selection using radio buttons
        self.model_type_group = QButtonGroup(self)
        
        self.use_novel_radio = QRadioButton("Novel Models")
        self.use_novel_radio.setToolTip("Use cutting-edge, state-of-the-art models (may be slower)")
        model_layout.addWidget(self.use_novel_radio)
        
        # Add description beneath the radio button
        novel_desc = QLabel("State-of-the-art models with higher quality results")
        novel_desc.setIndent(22)  # Indent to align with radio button
        novel_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        model_layout.addWidget(novel_desc)
        
        # Add some spacing
        spacer = QFrame()
        spacer.setFrameShape(QFrame.Shape.HLine)
        spacer.setFrameShadow(QFrame.Shadow.Sunken)
        model_layout.addWidget(spacer)
        
        self.use_foundational_radio = QRadioButton("Foundational Models")
        self.use_foundational_radio.setToolTip("Use established, proven models (faster processing)")
        model_layout.addWidget(self.use_foundational_radio)
        
        # Add description beneath the radio button
        found_desc = QLabel("Faster, more efficient models with reliable results")
        found_desc.setIndent(22)  # Indent to align with radio button
        found_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        model_layout.addWidget(found_desc)
        
        # Add radio buttons to group
        self.model_type_group.addButton(self.use_novel_radio)
        self.model_type_group.addButton(self.use_foundational_radio)
        
        # Connect the model type change
        self.model_type_group.buttonClicked.connect(self._on_model_type_changed)
        
        layout.addWidget(model_group)
        
        # Model selection group
        modules_group = QGroupBox("Enhancement Modules")
        modules_layout = QVBoxLayout(modules_group)
        
        # Denoising group
        self.enable_denoising = QCheckBox("Denoising")
        self.enable_denoising.setChecked(True)
        modules_layout.addWidget(self.enable_denoising)
        
        denoising_desc = QLabel("Reduces noise in images while preserving details")
        denoising_desc.setIndent(22)
        denoising_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        modules_layout.addWidget(denoising_desc)
        
        # Denoising strength slider
        denoising_control = QFrame()
        denoising_layout = QGridLayout(denoising_control)
        denoising_layout.setContentsMargins(22, 0, 0, 0)
        
        strength_label = QLabel("Strength:")
        denoising_layout.addWidget(strength_label, 0, 0)
        
        self.denoising_strength = QSlider(Qt.Orientation.Horizontal)
        self.denoising_strength.setRange(1, 100)
        self.denoising_strength.setValue(50)
        denoising_layout.addWidget(self.denoising_strength, 0, 1)
        
        self.denoising_strength_label = QLabel("50%")
        denoising_layout.addWidget(self.denoising_strength_label, 0, 2)
        
        # Connect slider to label
        self.denoising_strength.valueChanged.connect(
            lambda v: self.denoising_strength_label.setText(f"{v}%")
        )
        
        modules_layout.addWidget(denoising_control)
        
        # Add some spacing
        spacer = QFrame()
        spacer.setFrameShape(QFrame.Shape.HLine)
        spacer.setFrameShadow(QFrame.Shadow.Sunken)
        modules_layout.addWidget(spacer)
        
        # Super-resolution group
        self.enable_sr = QCheckBox("Super-Resolution")
        self.enable_sr.setChecked(True)
        modules_layout.addWidget(self.enable_sr)
        
        sr_desc = QLabel("Improves image resolution and clarity")
        sr_desc.setIndent(22)
        sr_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        modules_layout.addWidget(sr_desc)
        
        # Super-resolution controls
        sr_control = QFrame()
        sr_layout = QGridLayout(sr_control)
        sr_layout.setContentsMargins(22, 0, 0, 0)
        
        # Scale factor selection
        scale_label = QLabel("Scale Factor:")
        sr_layout.addWidget(scale_label, 0, 0)
        
        self.sr_scale = QComboBox()
        self.sr_scale.addItem("1x (Enhance Only)", 1)
        self.sr_scale.addItem("2x", 2)
        self.sr_scale.addItem("3x", 3)
        self.sr_scale.addItem("4x", 4)
        self.sr_scale.setCurrentIndex(1)  # Default to 2x
        sr_layout.addWidget(self.sr_scale, 0, 1, 1, 2)
        
        modules_layout.addWidget(sr_control)
        
        # Add some spacing
        spacer2 = QFrame()
        spacer2.setFrameShape(QFrame.Shape.HLine)
        spacer2.setFrameShadow(QFrame.Shadow.Sunken)
        modules_layout.addWidget(spacer2)
        
        # Artifact removal group
        self.enable_artifact = QCheckBox("Artifact Removal")
        self.enable_artifact.setChecked(True)
        modules_layout.addWidget(self.enable_artifact)
        
        artifact_desc = QLabel("Removes scanning artifacts and imperfections")
        artifact_desc.setIndent(22)
        artifact_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        modules_layout.addWidget(artifact_desc)
        
        # Artifact type controls
        artifact_control = QFrame()
        artifact_layout = QGridLayout(artifact_control)
        artifact_layout.setContentsMargins(22, 0, 0, 0)
        
        # Artifact type selection
        type_label = QLabel("Target Artifacts:")
        artifact_layout.addWidget(type_label, 0, 0)
        
        self.artifact_type = QComboBox()
        self.artifact_type.addItem("All Types", "all")
        self.artifact_type.addItem("Motion", "motion")
        self.artifact_type.addItem("Noise", "noise")
        self.artifact_type.addItem("Streaks", "streaks")
        artifact_layout.addWidget(self.artifact_type, 0, 1, 1, 2)
        
        modules_layout.addWidget(artifact_control)
        
        layout.addWidget(modules_group)
        
        # Add stretch to fill remaining space
        layout.addStretch()
    
    def _load_config(self):
        """Load settings from config."""
        # Set model type from config
        use_novel = self.config.get("models.use_novel", True)
        if use_novel:
            self.use_novel_radio.setChecked(True)
        else:
            self.use_foundational_radio.setChecked(True)
    
    def _on_model_type_changed(self, button):
        """Handle model type change."""
        use_novel = (button == self.use_novel_radio)
        self.config.set("models.use_novel", use_novel)
        self.config.save()
        
        logger.info(f"Model type changed to {'novel' if use_novel else 'foundational'}")
    
    def _on_clean_clicked(self):
        """Handle clean button click."""
        # Collect all options
        options = self.get_options()
        
        # Emit signal with options
        logger.debug(f"Cleaning requested with options: {options}")
        self.cleaningRequested.emit(options)
    
    def _reset_options(self):
        """Reset all options to defaults."""
        # Reset model type to default
        use_novel = self.config.get("models.use_novel", True)
        if use_novel:
            self.use_novel_radio.setChecked(True)
        else:
            self.use_foundational_radio.setChecked(True)
        
        # Reset module settings
        self.enable_denoising.setChecked(True)
        self.denoising_strength.setValue(50)
        
        self.enable_sr.setChecked(True)
        self.sr_scale.setCurrentIndex(1)  # 2x
        
        self.enable_artifact.setChecked(True)
        self.artifact_type.setCurrentIndex(0)  # All types
        
        # Reset processing settings
        self.processing_quality.setCurrentIndex(1)  # Standard
        self.process_color.setChecked(True)
        self.save_intermediates.setChecked(False)
        self.force_cpu.setChecked(False)
    
    def get_options(self):
        """
        Get the current cleaning options.
        
        Returns:
            dict: Dictionary of cleaning options
        """
        return {
            "use_novel_models": self.use_novel_radio.isChecked(),
            
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
                "quality": self.processing_quality.currentData(),
                "process_color": self.process_color.isChecked(),
                "save_intermediates": self.save_intermediates.isChecked(),
                "force_cpu": self.force_cpu.isChecked()
            }
        }
    
    def _init_settings_tab(self):
        """Initialize the settings tab content."""
        layout = QVBoxLayout(self.settings_tab)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(15)
        
        # Processing options group
        processing_group = QGroupBox("Processing Quality")
        processing_layout = QVBoxLayout(processing_group)
        
        # Processing quality selection
        quality_frame = QFrame()
        quality_layout = QHBoxLayout(quality_frame)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        
        quality_label = QLabel("Processing Quality:")
        quality_layout.addWidget(quality_label)
        
        self.processing_quality = QComboBox()
        self.processing_quality.addItem("Draft (Fast)", "draft")
        self.processing_quality.addItem("Standard", "standard")
        self.processing_quality.addItem("High (Slow)", "high")
        self.processing_quality.setCurrentIndex(1)  # Default to standard
        quality_layout.addWidget(self.processing_quality)
        
        processing_layout.addWidget(quality_frame)
        
        # Add description
        quality_desc = QLabel("Higher quality produces better results but takes longer")
        quality_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        processing_layout.addWidget(quality_desc)
        
        layout.addWidget(processing_group)
        
        # Advanced options group
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout(advanced_group)
        
        # Color mode option
        self.process_color = QCheckBox("Process Color Information")
        self.process_color.setChecked(True)
        self.process_color.setToolTip("When enabled, color information will be processed separately")
        advanced_layout.addWidget(self.process_color)
        
        # Save intermediates option
        self.save_intermediates = QCheckBox("Save Intermediate Results")
        self.save_intermediates.setChecked(False)
        self.save_intermediates.setToolTip("Save the output from each step in the enhancement pipeline")
        advanced_layout.addWidget(self.save_intermediates)
        
        # Use CPU if available
        self.force_cpu = QCheckBox("Force CPU Processing")
        self.force_cpu.setChecked(False)
        self.force_cpu.setToolTip("Use CPU even if GPU is available (slower but more compatible)")
        advanced_layout.addWidget(self.force_cpu)
        
        layout.addWidget(advanced_group)
        
        # Add stretch to fill remaining space
        layout.addStretch()