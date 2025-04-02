"""
Comparison view component for medical image enhancement application.
Provides side-by-side, overlay and split view comparison of original and enhanced images.
"""
import logging
from enum import Enum
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, pyqtSlot

from data.io.export import Exporter
from gui.viewers.image_viewer import ImageViewer

logger = logging.getLogger(__name__)

class ComparisonMode(Enum):
    """Comparison display modes."""
    SIDE_BY_SIDE = "side_by_side"
    OVERLAY = "overlay"
    SPLIT = "split"

class ComparisonView(QWidget):
    """
    Widget for comparing original and enhanced images.
    
    Features:
    - Side-by-side view of original and enhanced images
    - Overlay view with opacity control
    - Split view with movable divider
    - Export of comparison as a single image
    """
    
    def __init__(self, parent=None):
        """Initialize the comparison view."""
        super().__init__(parent)
        self.original_image = None
        self.enhanced_image = None
        self.comparison_mode = ComparisonMode.SIDE_BY_SIDE
        self.overlay_opacity = 0.5
        self.split_position = 0.5
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Comparison mode selector
        mode_label = QLabel("Mode:")
        controls_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Side by Side", ComparisonMode.SIDE_BY_SIDE.value)
        self.mode_combo.addItem("Overlay", ComparisonMode.OVERLAY.value)
        self.mode_combo.addItem("Split", ComparisonMode.SPLIT.value)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls_layout.addWidget(self.mode_combo)
        
        # Opacity slider for overlay mode
        self.opacity_label = QLabel("Opacity:")
        controls_layout.addWidget(self.opacity_label)
        
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        controls_layout.addWidget(self.opacity_slider)
        
        # Split position slider
        self.split_label = QLabel("Split:")
        controls_layout.addWidget(self.split_label)
        
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.valueChanged.connect(self._on_split_changed)
        controls_layout.addWidget(self.split_slider)
        
        # Add controls to main layout
        layout.addLayout(controls_layout)
        
        # Container for viewers
        self.viewers_container = QWidget()
        self.viewers_layout = QHBoxLayout(self.viewers_container)
        self.viewers_layout.setContentsMargins(0, 0, 0, 0)
        
        # Original image viewer
        self.original_viewer = ImageViewer()
        self.original_viewer.info_label.setText("Original")
        
        # Enhanced image viewer
        self.enhanced_viewer = ImageViewer()
        self.enhanced_viewer.info_label.setText("Enhanced")
        
        # Single viewer for overlay and split modes
        self.comparison_viewer = ImageViewer()
        
        # Initially set up side-by-side mode
        self.viewers_layout.addWidget(self.original_viewer)
        self.viewers_layout.addWidget(self.enhanced_viewer)
        
        # Add viewers container to main layout
        layout.addWidget(self.viewers_container)
        
        # Update UI for initial mode
        self._update_ui_for_mode()
    
    def set_images(self, original, enhanced, original_metadata=None, enhanced_metadata=None):
        """
        Set the original and enhanced images to compare.
        
        Args:
            original: Original image as numpy array
            enhanced: Enhanced image as numpy array
            original_metadata: Optional metadata for original image
            enhanced_metadata: Optional metadata for enhanced image
        """
        self.original_image = original
        self.enhanced_image = enhanced
        
        # Set images to individual viewers
        self.original_viewer.set_image(original, original_metadata)
        self.enhanced_viewer.set_image(enhanced, enhanced_metadata)
        
        # Update the comparison display
        self._update_comparison()
    
    def _update_comparison(self):
        """Update the comparison display based on current mode."""
        if self.original_image is None or self.enhanced_image is None:
            return
            
        if self.comparison_mode == ComparisonMode.SIDE_BY_SIDE:
            # Already displayed in separate viewers
            pass
        else:
            # Create a comparison image
            mode = self.comparison_mode.value
            if mode == ComparisonMode.OVERLAY.value:
                # Use the Exporter function to create overlay with current opacity
                alpha = self.overlay_opacity
                beta = 1.0 - alpha
                comparison = self.original_image * beta + self.enhanced_image * alpha
            elif mode == ComparisonMode.SPLIT.value:
                # Create a split view image
                comparison = self.original_image.copy()
                h, w = comparison.shape[:2]
                split_x = int(w * self.split_position)
                comparison[:, split_x:] = self.enhanced_image[:, split_x:]
            else:
                # Use side-by-side as fallback
                comparison = Exporter.create_comparison_image(
                    self.original_image, self.enhanced_image, mode='side_by_side'
                )
            
            # Display in the comparison viewer
            self.comparison_viewer.set_image(comparison)
    
    def _update_ui_for_mode(self):
        """Update UI based on the current comparison mode."""
        # Clear the viewers layout
        while self.viewers_layout.count():
            item = self.viewers_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        # Set up based on mode
        if self.comparison_mode == ComparisonMode.SIDE_BY_SIDE:
            self.viewers_layout.addWidget(self.original_viewer)
            self.viewers_layout.addWidget(self.enhanced_viewer)
            
            # Hide controls not applicable to this mode
            self.opacity_label.setVisible(False)
            self.opacity_slider.setVisible(False)
            self.split_label.setVisible(False)
            self.split_slider.setVisible(False)
        elif self.comparison_mode == ComparisonMode.OVERLAY:
            self.viewers_layout.addWidget(self.comparison_viewer)
            
            # Show opacity control, hide split control
            self.opacity_label.setVisible(True)
            self.opacity_slider.setVisible(True)
            self.split_label.setVisible(False)
            self.split_slider.setVisible(False)
        elif self.comparison_mode == ComparisonMode.SPLIT:
            self.viewers_layout.addWidget(self.comparison_viewer)
            
            # Show split control, hide opacity control
            self.opacity_label.setVisible(False)
            self.opacity_slider.setVisible(False)
            self.split_label.setVisible(True)
            self.split_slider.setVisible(True)
        
        # Update the comparison display
        self._update_comparison()
    
    @pyqtSlot(int)
    def _on_mode_changed(self, index):
        """Handle comparison mode changes."""
        mode_value = self.mode_combo.itemData(index)
        for mode in ComparisonMode:
            if mode.value == mode_value:
                self.comparison_mode = mode
                break
        
        self._update_ui_for_mode()
    
    @pyqtSlot(int)
    def _on_opacity_changed(self, value):
        """Handle opacity slider changes."""
        self.overlay_opacity = value / 100.0
        self._update_comparison()
    
    @pyqtSlot(int)
    def _on_split_changed(self, value):
        """Handle split slider changes."""
        self.split_position = value / 100.0
        self._update_comparison()
    
    def export_comparison(self, output_path=None):
        """
        Export the current comparison view to an image file.
        
        Args:
            output_path: Path to save the comparison image
            
        Returns:
            str: Path to saved file or None if cancelled/failed
        """
        if self.original_image is None or self.enhanced_image is None:
            logger.warning("Cannot export comparison: no images loaded")
            return None
            
        # Create comparison image based on current mode
        comparison = Exporter.create_comparison_image(
            self.original_image, 
            self.enhanced_image,
            output_path,
            mode=self.comparison_mode.value
        )
        
        return output_path if output_path else comparison