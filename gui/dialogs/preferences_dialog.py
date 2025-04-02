# Create a new file: gui/dialogs/preferences_dialog.py

"""
Preferences dialog for medical image enhancement application.
Allows users to configure application settings.
"""
import os
import logging
from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, 
    QFormLayout, QLabel, QComboBox, QLineEdit, QPushButton,
    QSpinBox, QCheckBox, QDialogButtonBox, QGroupBox, QFileDialog, 
    QWidget
)
from PyQt6.QtCore import Qt, QSettings, pyqtSlot
from PyQt6.QtGui import QIcon

from utils.config import Config

logger = logging.getLogger(__name__)

class PreferencesDialog(QDialog):
    """Dialog for configuring application preferences."""
    
    def __init__(self, parent=None):
        """Initialize the preferences dialog."""
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(500, 400)
        
        # Load configuration
        self.config = Config()
        
        # Initialize UI
        self._init_ui()
        
        # Load current settings
        self._load_settings()
    
    def _init_ui(self):
        """Initialize user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self._create_general_tab()
        self._create_processing_tab()
        self._create_models_tab()
        
        # Add tabs to widget
        layout.addWidget(self.tab_widget)
        
        # Add dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Reset
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self._reset_settings)
        
        layout.addWidget(button_box)
    
    def _create_general_tab(self):
        """Create the General tab."""
        general_tab = QWidget()
        layout = QVBoxLayout(general_tab)
        
        # Interface group
        interface_group = QGroupBox("Interface")
        form_layout = QFormLayout(interface_group)
        
        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Light")
        self.theme_combo.addItem("Dark")
        self.theme_combo.addItem("System")
        form_layout.addRow("Theme:", self.theme_combo)
        
        # Comparison view mode
        self.comparison_view_combo = QComboBox()
        self.comparison_view_combo.addItem("Side by Side")
        self.comparison_view_combo.addItem("Overlay")
        self.comparison_view_combo.addItem("Split")
        form_layout.addRow("Default Comparison View:", self.comparison_view_combo)
        
        layout.addWidget(interface_group)
        
        # Paths group
        paths_group = QGroupBox("Paths")
        form_layout = QFormLayout(paths_group)
        
        # Default save directory
        save_layout = QHBoxLayout()
        self.save_dir_edit = QLineEdit()
        save_layout.addWidget(self.save_dir_edit)
        self.browse_save_btn = QPushButton("Browse...")
        self.browse_save_btn.clicked.connect(self._browse_save_dir)
        save_layout.addWidget(self.browse_save_btn)
        form_layout.addRow("Default Save Directory:", save_layout)
        
        # Temp directory
        temp_layout = QHBoxLayout()
        self.temp_dir_edit = QLineEdit()
        temp_layout.addWidget(self.temp_dir_edit)
        self.browse_temp_btn = QPushButton("Browse...")
        self.browse_temp_btn.clicked.connect(self._browse_temp_dir)
        temp_layout.addWidget(self.browse_temp_btn)
        form_layout.addRow("Temp Directory:", temp_layout)
        
        layout.addWidget(paths_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(general_tab, "General")
    
    def _create_processing_tab(self):
        """Create the Processing tab."""
        processing_tab = QWidget()
        layout = QVBoxLayout(processing_tab)
        
        # Performance group
        performance_group = QGroupBox("Performance")
        form_layout = QFormLayout(performance_group)
        
        # Preview quality
        self.preview_quality_combo = QComboBox()
        self.preview_quality_combo.addItem("Low (Fast)")
        self.preview_quality_combo.addItem("Medium")
        self.preview_quality_combo.addItem("High (Slow)")
        form_layout.addRow("Preview Quality:", self.preview_quality_combo)
        
        # Threading option
        self.use_threading_check = QCheckBox("Use threading for processing")
        form_layout.addRow("Processing:", self.use_threading_check)
        
        # Max image size
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(512, 8192)
        self.max_size_spin.setSingleStep(512)
        self.max_size_spin.setSuffix(" pixels")
        form_layout.addRow("Max Image Size:", self.max_size_spin)
        
        layout.addWidget(performance_group)
        
        # Device group
        device_group = QGroupBox("Device")
        form_layout = QFormLayout(device_group)
        
        # Processing device
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (Best Available)")
        self.device_combo.addItem("CPU")
        self.device_combo.addItem("GPU (CUDA)")
        form_layout.addRow("Processing Device:", self.device_combo)
        
        layout.addWidget(device_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(processing_tab, "Processing")
    
    def _create_models_tab(self):
        """Create the Models tab."""
        models_tab = QWidget()
        layout = QVBoxLayout(models_tab)
        
        # Model weights group
        weights_group = QGroupBox("Model Weights")
        form_layout = QFormLayout(weights_group)
        
        # Denoising model
        dn_layout = QHBoxLayout()
        self.denoising_model_edit = QLineEdit()
        dn_layout.addWidget(self.denoising_model_edit)
        self.browse_dn_btn = QPushButton("Browse...")
        self.browse_dn_btn.clicked.connect(
            lambda: self._browse_model_weights("denoising")
        )
        dn_layout.addWidget(self.browse_dn_btn)
        form_layout.addRow("Denoising Model:", dn_layout)
        
        # Super-resolution model
        sr_layout = QHBoxLayout()
        self.sr_model_edit = QLineEdit()
        sr_layout.addWidget(self.sr_model_edit)
        self.browse_sr_btn = QPushButton("Browse...")
        self.browse_sr_btn.clicked.connect(
            lambda: self._browse_model_weights("super_resolution")
        )
        sr_layout.addWidget(self.browse_sr_btn)
        form_layout.addRow("Super-Resolution Model:", sr_layout)
        
        # Artifact removal model
        ar_layout = QHBoxLayout()
        self.artifact_model_edit = QLineEdit()
        ar_layout.addWidget(self.artifact_model_edit)
        self.browse_ar_btn = QPushButton("Browse...")
        self.browse_ar_btn.clicked.connect(
            lambda: self._browse_model_weights("artifact_removal")
        )
        ar_layout.addWidget(self.browse_ar_btn)
        form_layout.addRow("Artifact Removal Model:", ar_layout)
        
        layout.addWidget(weights_group)
        
        # Model type group
        type_group = QGroupBox("Default Model Type")
        type_layout = QVBoxLayout(type_group)
        
        # Novel vs Foundational
        self.use_novel_check = QCheckBox("Use novel models by default")
        type_layout.addWidget(self.use_novel_check)
        
        layout.addWidget(type_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(models_tab, "Models")
    
    def _load_settings(self):
        """Load current settings into the dialog."""
        # General tab
        self.theme_combo.setCurrentText(
            self.config.get("gui.theme", "Dark").capitalize()
        )
        
        self.comparison_view_combo.setCurrentText(
            self.config.get("gui.comparison_view", "Side by Side").replace("_", " ").title()
        )
        
        self.save_dir_edit.setText(
            self.config.get("paths.export_dir", "")
        )
        
        self.temp_dir_edit.setText(
            self.config.get("paths.temp_dir", "temp")
        )
        
        # Processing tab
        quality_map = {
            "low": "Low (Fast)",
            "medium": "Medium",
            "high": "High (Slow)"
        }
        self.preview_quality_combo.setCurrentText(
            quality_map.get(self.config.get("processing.preview_quality", "medium"), "Medium")
        )
        
        self.use_threading_check.setChecked(
            self.config.get("processing.use_threading", True)
        )
        
        self.max_size_spin.setValue(
            self.config.get("processing.max_image_dimension", 2048)
        )
        
        device_map = {
            "auto": "Auto (Best Available)",
            "cpu": "CPU",
            "cuda": "GPU (CUDA)"
        }
        self.device_combo.setCurrentText(
            device_map.get(self.config.get("models.denoising.device", "auto"), "Auto (Best Available)")
        )
        
        # Models tab
        model_type = "novel" if self.config.get("models.use_novel", True) else "foundational"
        
        self.denoising_model_edit.setText(
            self.config.get(f"models.denoising.{model_type}.model_path", "")
        )
        
        self.sr_model_edit.setText(
            self.config.get(f"models.super_resolution.{model_type}.model_path", "")
        )
        
        self.artifact_model_edit.setText(
            self.config.get(f"models.artifact_removal.{model_type}.model_path", "")
        )
        
        self.use_novel_check.setChecked(
            self.config.get("models.use_novel", True)
        )
    
    def accept(self):
        """Save settings and close dialog."""
        # Save settings
        
        # General tab
        self.config.set(
            "gui.theme", 
            self.theme_combo.currentText().lower()
        )
        
        self.config.set(
            "gui.comparison_view", 
            self.comparison_view_combo.currentText().replace(" ", "_").lower()
        )
        
        self.config.set(
            "paths.export_dir",
            self.save_dir_edit.text()
        )
        
        self.config.set(
            "paths.temp_dir",
            self.temp_dir_edit.text()
        )
        
        # Processing tab
        quality_map = {
            "Low (Fast)": "low",
            "Medium": "medium",
            "High (Slow)": "high"
        }
        self.config.set(
            "processing.preview_quality",
            quality_map.get(self.preview_quality_combo.currentText(), "medium")
        )
        
        self.config.set(
            "processing.use_threading",
            self.use_threading_check.isChecked()
        )
        
        self.config.set(
            "processing.max_image_dimension",
            self.max_size_spin.value()
        )
        
        device_map = {
            "Auto (Best Available)": "auto",
            "CPU": "cpu",
            "GPU (CUDA)": "cuda"
        }
        device = device_map.get(self.device_combo.currentText(), "auto")
        self.config.set("models.denoising.device", device)
        self.config.set("models.super_resolution.device", device)
        self.config.set("models.artifact_removal.device", device)
        
        # Models tab
        model_type = "novel" if self.use_novel_check.isChecked() else "foundational"
        self.config.set("models.use_novel", self.use_novel_check.isChecked())
        
        self.config.set(
            f"models.denoising.{model_type}.model_path",
            self.denoising_model_edit.text()
        )
        
        self.config.set(
            f"models.super_resolution.{model_type}.model_path",
            self.sr_model_edit.text()
        )
        
        self.config.set(
            f"models.artifact_removal.{model_type}.model_path",
            self.artifact_model_edit.text()
        )
        
        # Save configuration
        self.config.save()
        
        # Close dialog
        super().accept()
    
    def _reset_settings(self):
        """Reset settings to defaults."""
        # Reset configuration
        self.config = Config()
        
        # Reload settings
        self._load_settings()
    
    def _browse_save_dir(self):
        """Browse for default save directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Default Save Directory", self.save_dir_edit.text()
        )
        if dir_path:
            self.save_dir_edit.setText(dir_path)
    
    def _browse_temp_dir(self):
        """Browse for temp directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Temp Directory", self.temp_dir_edit.text()
        )
        if dir_path:
            self.temp_dir_edit.setText(dir_path)
    
    def _browse_model_weights(self, model_type):
        """
        Browse for model weights file.
        
        Args:
            model_type: Type of model (denoising, super_resolution, artifact_removal)
        """
        # Determine which line edit to update
        if model_type == "denoising":
            edit = self.denoising_model_edit
        elif model_type == "super_resolution":
            edit = self.sr_model_edit
        elif model_type == "artifact_removal":
            edit = self.artifact_model_edit
        else:
            return
        
        # Show file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {model_type.replace('_', ' ').title()} Model Weights",
            edit.text(), "Model Files (*.pth *.pt *.bin);;All Files (*)"
        )
        
        if file_path:
            edit.setText(file_path)