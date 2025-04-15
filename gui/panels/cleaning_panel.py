"""
Cleaning panel component for medical image enhancement application.
Provides controls for image enhancement operations with improved model selection.
"""
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QCheckBox, QComboBox, QLabel, QSlider, QPushButton,
    QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
    QFrame, QTabWidget, QGridLayout, QSizePolicy, QFormLayout,
    QScrollArea, QToolTip
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon, QFont, QCursor
from PyQt6.QtGui import QIcon, QFont, QCursor, QColor

from utils.config import Config
from utils.model_service import ModelService

logger = logging.getLogger(__name__)

class CleaningPanel(QWidget):
    """
    Control panel for image cleaning and enhancement options.
    
    Features:
    - Selection of specific models for each enhancement type
    - Adjustable parameters for each cleaning method
    - Visual feedback for model availability
    - Detailed tooltips with model information
    """
    
    # Signals
    cleaningRequested = pyqtSignal(dict)  # Emitted when clean button is pressed
    
    def __init__(self, parent=None):
        """Initialize the cleaning panel."""
        super().__init__(parent)
        self.config = Config()
        
        # Create model service
        self.model_service = ModelService(self.config)
        
        self._init_ui()
        self._load_config()
        self._populate_model_dropdowns()
    
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
        
        # Create scroll area for settings to handle small screens
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        scroll_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.models_tab = QWidget()
        self.settings_tab = QWidget()
        
        self.tab_widget.addTab(self.models_tab, "Models")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        
        # Initialize tab contents
        self._init_models_tab()
        self._init_settings_tab()
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
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
        
        # Denoising model group
        denoising_group = QGroupBox("Denoising Models")
        denoising_layout = QFormLayout(denoising_group)
        
        self.enable_denoising = QCheckBox("Enable Denoising")
        self.enable_denoising.setChecked(True)
        self.enable_denoising.stateChanged.connect(self._update_ui_state)
        denoising_layout.addRow(self.enable_denoising, QLabel())
        
        denoising_desc = QLabel("Reduces noise in images while preserving details")
        denoising_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        denoising_layout.addRow("", denoising_desc)
        
        # Model selection dropdown
        self.denoising_model = QComboBox()
        self.denoising_model.setToolTip("Select the denoising model to use")
        denoising_layout.addRow("Model:", self.denoising_model)
        
        # Denoising strength slider
        strength_layout = QHBoxLayout()
        self.denoising_strength = QSlider(Qt.Orientation.Horizontal)
        self.denoising_strength.setRange(1, 100)
        self.denoising_strength.setValue(50)
        strength_layout.addWidget(self.denoising_strength, 1)
        
        self.denoising_strength_label = QLabel("50%")
        strength_layout.addWidget(self.denoising_strength_label)
        
        # Connect slider to label
        self.denoising_strength.valueChanged.connect(
            lambda v: self.denoising_strength_label.setText(f"{v}%")
        )
        
        denoising_layout.addRow("Strength:", strength_layout)
        
        layout.addWidget(denoising_group)
        
        # Super-resolution model group
        sr_group = QGroupBox("Super-Resolution Models")
        sr_layout = QFormLayout(sr_group)
        
        self.enable_sr = QCheckBox("Enable Super-Resolution")
        self.enable_sr.setChecked(True)
        self.enable_sr.stateChanged.connect(self._update_ui_state)
        sr_layout.addRow(self.enable_sr, QLabel())
        
        sr_desc = QLabel("Improves image resolution and clarity")
        sr_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        sr_layout.addRow("", sr_desc)
        
        # Model selection dropdown
        self.sr_model = QComboBox()
        self.sr_model.setToolTip("Select the super-resolution model to use")
        sr_layout.addRow("Model:", self.sr_model)
        
        # Scale factor selection
        self.sr_scale = QComboBox()
        self.sr_scale.addItem("1x (Enhance Only)", 1)
        self.sr_scale.addItem("2x", 2)
        self.sr_scale.addItem("4x", 4)
        self.sr_scale.addItem("8x", 8)
        self.sr_scale.setCurrentIndex(1)  # Default to 2x
        self.sr_scale.setToolTip("Select the scale factor for super-resolution")
        sr_layout.addRow("Scale Factor:", self.sr_scale)
        
        layout.addWidget(sr_group)
        
        # Artifact removal model group
        artifact_group = QGroupBox("Artifact Removal Models")
        artifact_layout = QFormLayout(artifact_group)
        
        self.enable_artifact = QCheckBox("Enable Artifact Removal")
        self.enable_artifact.setChecked(True)
        self.enable_artifact.stateChanged.connect(self._update_ui_state)
        artifact_layout.addRow(self.enable_artifact, QLabel())
        
        artifact_desc = QLabel("Removes scanning artifacts and imperfections")
        artifact_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        artifact_layout.addRow("", artifact_desc)
        
        # Model selection dropdown
        self.artifact_model = QComboBox()
        self.artifact_model.setToolTip("Select the artifact removal model to use")
        artifact_layout.addRow("Model:", self.artifact_model)
        
        # Artifact type selection
        self.artifact_type = QComboBox()
        self.artifact_type.addItem("All Types", "all")
        self.artifact_type.addItem("Motion", "motion")
        self.artifact_type.addItem("Noise", "noise")
        self.artifact_type.addItem("Streaks", "streaks")
        self.artifact_type.setToolTip("Select which types of artifacts to target")
        artifact_layout.addRow("Target Artifacts:", self.artifact_type)
        
        layout.addWidget(artifact_group)
        
        # Add stretch to fill remaining space
        layout.addStretch()
    
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
    
    def _populate_model_dropdowns(self):
        """Populate the model selection dropdowns."""
        try:
            # Get available models from the model service
            denoising_models = self.model_service.get_available_models(model_type="denoising")
            sr_models = self.model_service.get_available_models(model_type="super_resolution")
            artifact_models = self.model_service.get_available_models(model_type="artifact_removal")
            enhancement_models = self.model_service.get_available_models(model_type="enhancement")
            
            # Clear existing items
            self.denoising_model.clear()
            self.sr_model.clear()
            self.artifact_model.clear()
            
            # Add denoising models
            for model in denoising_models:
                # Mark available/unavailable in the display name
                available_mark = "✓ " if model["available"] else "✗ "
                display_name = f"{available_mark}{model['description']}"
                
                # Add detailed tooltip
                tooltip = f"Model: {model['id']}\nCategory: {model['category']}\nAvailable: {'Yes' if model['available'] else 'No'}"
                if model["path"]:
                    tooltip += f"\nPath: {model['path']}"
                
                # Add to dropdown
                self.denoising_model.addItem(display_name, model['id'])
                self.denoising_model.setItemData(self.denoising_model.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
                
                # If not available, make the text gray
                if not model["available"]:
                    self.denoising_model.setItemData(
                        self.denoising_model.count() - 1, 
                        QColor("#666666"), 
                        Qt.ItemDataRole.ForegroundRole
                    )
            
            # Add enhancement models to denoising dropdown (as they can be used for enhancement too)
            for model in enhancement_models:
                # Only add if available
                if model["available"]:
                    display_name = f"✓ {model['description']} (Enhancement)"
                    tooltip = f"Model: {model['id']}\nCategory: {model['category']}\nType: Enhancement model used for denoising"
                    
                    self.denoising_model.addItem(display_name, model['id'])
                    self.denoising_model.setItemData(self.denoising_model.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
            
            # Add super-resolution models
            for model in sr_models:
                available_mark = "✓ " if model["available"] else "✗ "
                display_name = f"{available_mark}{model['description']}"
                
                tooltip = f"Model: {model['id']}\nCategory: {model['category']}\nAvailable: {'Yes' if model['available'] else 'No'}"
                if model["path"]:
                    tooltip += f"\nPath: {model['path']}"
                
                self.sr_model.addItem(display_name, model['id'])
                self.sr_model.setItemData(self.sr_model.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
                
                if not model["available"]:
                    self.sr_model.setItemData(
                        self.sr_model.count() - 1, 
                        QColor("#666666"), 
                        Qt.ItemDataRole.ForegroundRole
                    )
            
            # Add artifact removal models
            for model in artifact_models:
                available_mark = "✓ " if model["available"] else "✗ "
                display_name = f"{available_mark}{model['description']}"
                
                tooltip = f"Model: {model['id']}\nCategory: {model['category']}\nAvailable: {'Yes' if model['available'] else 'No'}"
                if model["path"]:
                    tooltip += f"\nPath: {model['path']}"
                
                self.artifact_model.addItem(display_name, model['id'])
                self.artifact_model.setItemData(self.artifact_model.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
                
                if not model["available"]:
                    self.artifact_model.setItemData(
                        self.artifact_model.count() - 1, 
                        QColor("#666666"), 
                        Qt.ItemDataRole.ForegroundRole
                    )
            
            # Select default models based on current preference (novel/foundational)
            self._update_default_model_selections()
            
        except Exception as e:
            logger.error(f"Error populating model dropdowns: {e}")
    
    def _update_default_model_selections(self):
        """Update the default selected models based on novel/foundational preference."""
        try:
            use_novel = self.use_novel_radio.isChecked()
            
            # Set defaults for denoising
            if use_novel:
                # Try to find a novel model in order of preference
                novel_denoising_models = ["novel_vit_mae_cxr", "novel_resnet50_rad", 
                                          "novel_resnet50_medical", "novel_swinvit", 
                                          "novel_diffusion_denoiser"]
                
                for model_id in novel_denoising_models:
                    index = self.denoising_model.findData(model_id)
                    if index >= 0:
                        self.denoising_model.setCurrentIndex(index)
                        break
            else:
                # Find the foundational denoising model
                index = self.denoising_model.findData("dncnn_denoiser")
                if index >= 0:
                    self.denoising_model.setCurrentIndex(index)
            
            # Set defaults for super-resolution
            if use_novel:
                novel_sr_models = ["novel_restormer", "novel_swinir_super_resolution"]
                for model_id in novel_sr_models:
                    index = self.sr_model.findData(model_id)
                    if index >= 0:
                        self.sr_model.setCurrentIndex(index)
                        break
            else:
                index = self.sr_model.findData("edsr_super_resolution")
                if index >= 0:
                    self.sr_model.setCurrentIndex(index)
            
            # Set defaults for artifact removal
            if use_novel:
                index = self.artifact_model.findData("novel_stylegan_artifact_removal")
                if index >= 0:
                    self.artifact_model.setCurrentIndex(index)
            else:
                index = self.artifact_model.findData("unet_artifact_removal")
                if index >= 0:
                    self.artifact_model.setCurrentIndex(index)
            
        except Exception as e:
            logger.error(f"Error setting default models: {e}")
    
    def _update_ui_state(self):
        """Update the UI state based on checkbox values."""
        # Enable/disable denoising controls
        denoising_enabled = self.enable_denoising.isChecked()
        self.denoising_model.setEnabled(denoising_enabled)
        self.denoising_strength.setEnabled(denoising_enabled)
        self.denoising_strength_label.setEnabled(denoising_enabled)
        
        # Enable/disable super-resolution controls
        sr_enabled = self.enable_sr.isChecked()
        self.sr_model.setEnabled(sr_enabled)
        self.sr_scale.setEnabled(sr_enabled)
        
        # Enable/disable artifact removal controls
        artifact_enabled = self.enable_artifact.isChecked()
        self.artifact_model.setEnabled(artifact_enabled)
        self.artifact_type.setEnabled(artifact_enabled)
    
    def _load_config(self):
        """Load settings from config."""
        # Set model type from config
        use_novel = self.config.get("models.use_novel", True)
        if use_novel:
            self.use_novel_radio.setChecked(True)
        else:
            self.use_foundational_radio.setChecked(True)
        
        # Initialize UI state
        self._update_ui_state()
    
    def _on_model_type_changed(self, button):
        """Handle model type change."""
        use_novel = (button == self.use_novel_radio)
        self.config.set("models.use_novel", use_novel)
        self.config.save()
        
        logger.info(f"Model type changed to {'novel' if use_novel else 'foundational'}")
        
        # Update default model selections
        self._update_default_model_selections()
    
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
        
        # Update default model selections
        self._update_default_model_selections()
        
        # Update UI state
        self._update_ui_state()
    
    def get_options(self):
        """
        Get the current cleaning options.
        
        Returns:
            dict: Dictionary of cleaning options
        """
        # Get selected model IDs
        denoising_model = self.denoising_model.currentData()
        sr_model = self.sr_model.currentData()
        artifact_model = self.artifact_model.currentData()
        
        return {
            "use_novel_models": self.use_novel_radio.isChecked(),
            
            "denoising": {
                "enabled": self.enable_denoising.isChecked(),
                "model_id": denoising_model,
                "strength": self.denoising_strength.value() / 100.0
            },
            
            "super_resolution": {
                "enabled": self.enable_sr.isChecked(),
                "model_id": sr_model,
                "scale_factor": self.sr_scale.currentData()
            },
            
            "artifact_removal": {
                "enabled": self.enable_artifact.isChecked(),
                "model_id": artifact_model,
                "artifact_type": self.artifact_type.currentData()
            },
            
            "processing": {
                "quality": self.processing_quality.currentData(),
                "process_color": self.process_color.isChecked(),
                "save_intermediates": self.save_intermediates.isChecked(),
                "force_cpu": self.force_cpu.isChecked()
            }
        }