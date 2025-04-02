# File: gui/viewers/segmentation_view.py

"""
Segmentation view component for medical image enhancement application.
Displays segmentation masks with overlay options.
"""
import logging
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QPushButton, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from gui.viewers.image_viewer import ImageViewer

logger = logging.getLogger(__name__)

class SegmentationView(QWidget):
    """
    Widget for displaying segmentation masks overlay on images.
    
    Features:
    - Display segmentation masks as colored overlays
    - Control opacity of the overlay
    - Select which segments to display
    """
    
    def __init__(self, parent=None):
        """Initialize the segmentation view."""
        super().__init__(parent)
        self.image_data = None
        self.mask_data = None
        self.class_colors = [
            (255, 0, 0, 128),    # Red (semi-transparent)
            (0, 255, 0, 128),    # Green (semi-transparent)
            (0, 0, 255, 128),    # Blue (semi-transparent)
            (255, 255, 0, 128),  # Yellow (semi-transparent)
            (255, 0, 255, 128),  # Magenta (semi-transparent)
            (0, 255, 255, 128),  # Cyan (semi-transparent)
        ]
        self.class_names = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
        self.overlay_opacity = 0.5
        self.visible_classes = set(range(len(self.class_names)))
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Opacity slider
        opacity_layout = QVBoxLayout()
        opacity_layout.addWidget(QLabel("Overlay Opacity:"))
        
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.overlay_opacity * 100))
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        
        controls_layout.addLayout(opacity_layout)
        
        # Segmentation class selector
        class_layout = QVBoxLayout()
        class_layout.addWidget(QLabel("Visible Classes:"))
        
        self.class_checkboxes = []
        for i, name in enumerate(self.class_names):
            checkbox = QCheckBox(name)
            checkbox.setChecked(i in self.visible_classes)
            checkbox.stateChanged.connect(lambda state, idx=i: self._on_class_toggled(idx, state))
            class_layout.addWidget(checkbox)
            self.class_checkboxes.append(checkbox)
        
        controls_layout.addLayout(class_layout)
        
        # Add controls to main layout
        layout.addLayout(controls_layout)
        
        # Image viewer
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer)
        
        # Initial update
        self._update_display()
    
    def _on_opacity_changed(self, value):
        """Handle opacity slider changes."""
        self.overlay_opacity = value / 100.0
        self._update_display()
    
    def _on_class_toggled(self, class_idx, checked):
        """Handle class visibility toggling."""
        if checked:
            self.visible_classes.add(class_idx)
        else:
            self.visible_classes.discard(class_idx)
        self._update_display()
    
    def set_image_and_mask(self, image_data, mask_data, class_names=None):
        """
        Set the image and segmentation mask to display.
        
        Args:
            image_data: Image as numpy array
            mask_data: Segmentation mask as numpy array (class indices)
            class_names: Optional list of class names
        """
        self.image_data = image_data
        self.mask_data = mask_data
        
        if class_names:
            self.class_names = class_names
            # Update checkboxes
            for i, name in enumerate(self.class_names):
                if i < len(self.class_checkboxes):
                    self.class_checkboxes[i].setText(name)
                    self.class_checkboxes[i].setVisible(True)
            
            # Hide extra checkboxes
            for i in range(len(self.class_names), len(self.class_checkboxes)):
                self.class_checkboxes[i].setVisible(False)
        
        self._update_display()
    
    def _update_display(self):
        """Update the display with current image and mask."""
        if self.image_data is None:
            return
            
        # If mask is not available, just show the image
        if self.mask_data is None:
            self.image_viewer.set_image(self.image_data)
            return
        
        # Create a copy of the image for overlay
        if np.issubdtype(self.image_data.dtype, np.floating):
            display_image = (self.image_data * 255).clip(0, 255).astype(np.uint8)
        else:
            display_image = self.image_data.copy()
        
        # Convert to RGB if grayscale
        if len(display_image.shape) == 2:
            display_image = np.stack([display_image] * 3, axis=2)
        
        # Add alpha channel if needed
        if display_image.shape[2] == 3:
            display_image = np.concatenate(
                [display_image, np.full((display_image.shape[0], display_image.shape[1], 1), 255, dtype=np.uint8)],
                axis=2
            )
        
        # Apply mask overlay
        for class_idx in self.visible_classes:
            if class_idx >= len(self.class_colors):
                continue
                
            # Get color for this class
            color = self.class_colors[class_idx]
            
            # Find pixels of this class
            mask = self.mask_data == class_idx
            
            if np.any(mask):
                # Apply color to pixels where mask is True
                for c in range(3):
                    display_image[mask, c] = int(
                        display_image[mask, c] * (1 - self.overlay_opacity) + 
                        color[c] * self.overlay_opacity
                    )
        
        # Display the result
        self.image_viewer.set_image(display_image)