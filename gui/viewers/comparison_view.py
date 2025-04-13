"""
Comparison view component for medical image enhancement application.
Provides professional side-by-side, overlay and split view comparison of original and enhanced images.
"""
import logging
from enum import Enum
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QFrame, QPushButton, QGridLayout, QGroupBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, pyqtSlot, QSize

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
    Widget for comparing original and enhanced images with a professional medical design.
    
    Features:
    - Side-by-side view of original and enhanced images
    - Overlay view with opacity control
    - Split view with movable divider
    - Export of comparison as a single image
    - Synchronized zooming and panning
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title label
        title_label = QLabel("Image Comparison")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Controls group
        controls_group = QGroupBox("View Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Comparison mode selector
        mode_label = QLabel("Mode:")
        controls_layout.addWidget(mode_label, 0, 0)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Side by Side", ComparisonMode.SIDE_BY_SIDE.value)
        self.mode_combo.addItem("Overlay", ComparisonMode.OVERLAY.value)
        self.mode_combo.addItem("Split", ComparisonMode.SPLIT.value)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls_layout.addWidget(self.mode_combo, 0, 1)
        
        # Opacity slider for overlay mode
        self.opacity_label = QLabel("Opacity:")
        controls_layout.addWidget(self.opacity_label, 0, 2)
        
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        controls_layout.addWidget(self.opacity_slider, 0, 3)
        
        # Split position slider
        self.split_label = QLabel("Split:")
        controls_layout.addWidget(self.split_label, 0, 4)
        
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.valueChanged.connect(self._on_split_changed)
        controls_layout.addWidget(self.split_slider, 0, 5)
        
        # Add Export button
        self.export_button = QPushButton("Export Comparison")
        self.export_button.clicked.connect(self._on_export_clicked)
        controls_layout.addWidget(self.export_button, 0, 6)
        
        # Add controls to main layout
        layout.addWidget(controls_group)
        
        # Container for viewers
        self.viewers_container = QWidget()
        self.viewers_layout = QHBoxLayout(self.viewers_container)
        self.viewers_layout.setContentsMargins(0, 0, 0, 0)
        self.viewers_layout.setSpacing(10)
        
        # Original image viewer
        self.original_viewer = ImageViewer()
        self.original_viewer.info_label.setText("Original")
        self.original_viewer.info_label.setStyleSheet("color: #333333; font-weight: bold;")
        
        # Enhanced image viewer
        self.enhanced_viewer = ImageViewer()
        self.enhanced_viewer.info_label.setText("Enhanced")
        self.enhanced_viewer.info_label.setStyleSheet("color: #0078D7; font-weight: bold;")
        
        # Single viewer for overlay and split modes
        self.comparison_viewer = ImageViewer()
        self.comparison_viewer.info_label.setText("Comparison")
        self.comparison_viewer.info_label.setStyleSheet("color: #0078D7; font-weight: bold;")
        
        # Add frames around viewers for visual separation
        self.original_frame = QFrame()
        self.original_frame.setFrameShape(QFrame.Shape.Box)
        self.original_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.original_frame.setLineWidth(1)
        self.original_frame.setMidLineWidth(0)
        
        original_frame_layout = QVBoxLayout(self.original_frame)
        original_frame_layout.setContentsMargins(0, 0, 0, 0)
        original_frame_layout.addWidget(self.original_viewer)
        
        self.enhanced_frame = QFrame()
        self.enhanced_frame.setFrameShape(QFrame.Shape.Box)
        self.enhanced_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.enhanced_frame.setLineWidth(1)
        self.enhanced_frame.setMidLineWidth(0)
        
        enhanced_frame_layout = QVBoxLayout(self.enhanced_frame)
        enhanced_frame_layout.setContentsMargins(0, 0, 0, 0)
        enhanced_frame_layout.addWidget(self.enhanced_viewer)
        
        self.comparison_frame = QFrame()
        self.comparison_frame.setFrameShape(QFrame.Shape.Box)
        self.comparison_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.comparison_frame.setLineWidth(1)
        self.comparison_frame.setMidLineWidth(0)
        
        comparison_frame_layout = QVBoxLayout(self.comparison_frame)
        comparison_frame_layout.setContentsMargins(0, 0, 0, 0)
        comparison_frame_layout.addWidget(self.comparison_viewer)
        
        # Initially set up side-by-side mode
        self.viewers_layout.addWidget(self.original_frame)
        self.viewers_layout.addWidget(self.enhanced_frame)
        
        # Add viewers container to main layout
        layout.addWidget(self.viewers_container)
        
        # Update UI for initial mode
        self._update_ui_for_mode()
        
        # Connect viewer signals for synchronized scrolling
        self.original_viewer.graphics_view.horizontalScrollBar().valueChanged.connect(
            self.sync_horizontal_scroll
        )
        self.original_viewer.graphics_view.verticalScrollBar().valueChanged.connect(
            self.sync_vertical_scroll
        )
        self.enhanced_viewer.graphics_view.horizontalScrollBar().valueChanged.connect(
            self.sync_horizontal_scroll
        )
        self.enhanced_viewer.graphics_view.verticalScrollBar().valueChanged.connect(
            self.sync_vertical_scroll
        )
        
        # Connect viewer zoom signals
        self.original_viewer.zoomChanged.connect(
            lambda factor: self.sync_zoom(factor, self.original_viewer)
        )
        self.enhanced_viewer.zoomChanged.connect(
            lambda factor: self.sync_zoom(factor, self.enhanced_viewer)
        )
    
    def sync_horizontal_scroll(self, value):
        """Synchronize horizontal scrolling between viewers."""
        if self.comparison_mode == ComparisonMode.SIDE_BY_SIDE:
            sender = self.sender()
            if sender == self.original_viewer.graphics_view.horizontalScrollBar():
                self.enhanced_viewer.graphics_view.horizontalScrollBar().setValue(value)
            elif sender == self.enhanced_viewer.graphics_view.horizontalScrollBar():
                self.original_viewer.graphics_view.horizontalScrollBar().setValue(value)

    def sync_vertical_scroll(self, value):
        """Synchronize vertical scrolling between viewers."""
        if self.comparison_mode == ComparisonMode.SIDE_BY_SIDE:
            sender = self.sender()
            if sender == self.original_viewer.graphics_view.verticalScrollBar():
                self.enhanced_viewer.graphics_view.verticalScrollBar().setValue(value)
            elif sender == self.enhanced_viewer.graphics_view.verticalScrollBar():
                self.original_viewer.graphics_view.verticalScrollBar().setValue(value)

    def sync_zoom(self, factor, viewer):
        """Synchronize zoom between viewers."""
        if self.comparison_mode == ComparisonMode.SIDE_BY_SIDE:
            if viewer == self.original_viewer:
                self.enhanced_viewer.zoom_factor = factor
                self.enhanced_viewer._update_zoom_label()
            else:
                self.original_viewer.zoom_factor = factor
                self.original_viewer._update_zoom_label()
    
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
        
        # Enable export button if we have both images
        self.export_button.setEnabled(self.original_image is not None and 
                                      self.enhanced_image is not None and
                                      id(self.original_image) != id(self.enhanced_image))
    
    def _update_comparison(self):
            """Update the comparison display based on current mode."""
            if self.original_image is None or self.enhanced_image is None:
                return
                
            try:
                if self.comparison_mode == ComparisonMode.SIDE_BY_SIDE:
                    # Already displayed in separate viewers
                    pass
                else:
                    # Create a comparison image
                    mode = self.comparison_mode.value
                    
                    # Handle images with different shapes
                    if self.original_image.shape != self.enhanced_image.shape:
                        logger.warning(f"Image shapes don't match: original {self.original_image.shape}, enhanced {self.enhanced_image.shape}")
                        # Resize enhanced to match original
                        from data.processing.transforms import resize_image
                        enhanced_resized = resize_image(
                            self.enhanced_image, 
                            (self.original_image.shape[1], self.original_image.shape[0]), 
                            preserve_aspect_ratio=False
                        )
                    else:
                        enhanced_resized = self.enhanced_image
                    
                    if mode == ComparisonMode.OVERLAY.value:
                        # Use the Exporter function to create overlay with current opacity
                        try:
                            alpha = self.overlay_opacity
                            beta = 1.0 - alpha
                            comparison = self.original_image * beta + enhanced_resized * alpha
                        except Exception as e:
                            logger.error(f"Error creating overlay: {e}")
                            # Fallback to simple overlay
                            comparison = enhanced_resized.copy()
                            
                    elif mode == ComparisonMode.SPLIT.value:
                        # Create a split view image
                        try:
                            comparison = self.original_image.copy()
                            h, w = comparison.shape[:2]
                            split_x = int(w * self.split_position)
                            
                            # Handle different number of channels
                            if len(comparison.shape) != len(enhanced_resized.shape):
                                if len(comparison.shape) == 2 and len(enhanced_resized.shape) == 3:
                                    # Convert comparison to RGB
                                    comparison = np.stack([comparison] * 3, axis=2)
                                elif len(comparison.shape) == 3 and len(enhanced_resized.shape) == 2:
                                    # Convert enhanced to RGB
                                    enhanced_resized = np.stack([enhanced_resized] * 3, axis=2)
                            
                            comparison[:, split_x:] = enhanced_resized[:, split_x:]
                            
                            # Add a vertical line at the split point
                            line_width = 2
                            split_range = max(1, min(split_x + line_width, w) - max(0, split_x - line_width))
                            
                            # Create a bright line that will be visible on any background
                            if len(comparison.shape) == 2:  # Grayscale
                                comparison[:, max(0, split_x - line_width):min(w, split_x + line_width)] = 1.0
                            else:  # Color
                                # Use blue for the split line
                                comparison[:, max(0, split_x - line_width):min(w, split_x + line_width), 0] = 0.0  # R
                                comparison[:, max(0, split_x - line_width):min(w, split_x + line_width), 1] = 0.5  # G
                                comparison[:, max(0, split_x - line_width):min(w, split_x + line_width), 2] = 1.0  # B
                        except Exception as e:
                            logger.error(f"Error creating split view: {e}")
                            # Fallback to side-by-side
                            comparison = Exporter.create_comparison_image(
                                self.original_image, enhanced_resized, mode='side_by_side'
                            )
                    else:
                        # Use side-by-side as fallback
                        comparison = Exporter.create_comparison_image(
                            self.original_image, enhanced_resized, mode='side_by_side'
                        )
                    
                    # Display in the comparison viewer
                    self.comparison_viewer.set_image(comparison)
            except Exception as e:
                logger.error(f"Error updating comparison: {e}")
                # If all else fails, just show the enhanced image
                self.comparison_viewer.set_image(self.enhanced_image)
    
    def _update_ui_for_mode(self):
        """Update UI based on the current comparison mode."""
        # Clear the viewers layout
        while self.viewers_layout.count():
            item = self.viewers_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        # Set up based on mode
        if self.comparison_mode == ComparisonMode.SIDE_BY_SIDE:
            self.viewers_layout.addWidget(self.original_frame)
            self.viewers_layout.addWidget(self.enhanced_frame)
            
            # Hide controls not applicable to this mode
            self.opacity_label.setVisible(False)
            self.opacity_slider.setVisible(False)
            self.split_label.setVisible(False)
            self.split_slider.setVisible(False)
        elif self.comparison_mode == ComparisonMode.OVERLAY:
            self.viewers_layout.addWidget(self.comparison_frame)
            
            # Show opacity control, hide split control
            self.opacity_label.setVisible(True)
            self.opacity_slider.setVisible(True)
            self.split_label.setVisible(False)
            self.split_slider.setVisible(False)
        elif self.comparison_mode == ComparisonMode.SPLIT:
            self.viewers_layout.addWidget(self.comparison_frame)
            
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
    
    def _on_export_clicked(self):
        """Handle export button clicks."""
        self.export_comparison()
    
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
        
        if output_path is None:
            # Let the caller handle the file dialog if no path provided
            return Exporter.create_comparison_image(
                self.original_image, 
                self.enhanced_image,
                None,
                mode=self.comparison_mode.value
            )
        else:
            # Create and save the comparison image
            return Exporter.create_comparison_image(
                self.original_image, 
                self.enhanced_image,
                output_path,
                mode=self.comparison_mode.value
            )