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
    QFrame, QPushButton, QGridLayout, QGroupBox, QSizePolicy,
    QCheckBox, QSplitter, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, pyqtSlot, QSize

from data.io.export import Exporter
from gui.viewers.image_viewer import ImageViewer
from gui.viewers.histogram_widget import HistogramWidget

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
    - Histogram comparison for quantitative analysis
    """
    
    def __init__(self, parent=None):
        """Initialize the comparison view."""
        super().__init__(parent)
        self.original_image = None
        self.enhanced_image = None
        self.comparison_mode = ComparisonMode.SIDE_BY_SIDE
        self.overlay_opacity = 0.5
        self.split_position = 0.5
        self.show_histograms = True
        
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
        
        # Add histogram toggle
        self.histogram_check = QCheckBox("Show Histograms")
        self.histogram_check.setChecked(True)
        self.histogram_check.stateChanged.connect(self._on_histogram_toggled)
        controls_layout.addWidget(self.histogram_check, 0, 6)
        
        # Add Export button
        self.export_button = QPushButton("Export Comparison")
        self.export_button.clicked.connect(self._on_export_clicked)
        controls_layout.addWidget(self.export_button, 0, 7)
        
        # Add controls to main layout
        layout.addWidget(controls_group)
        
        # Create a splitter for viewers and histograms
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        
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
        
        # Add viewers container to splitter
        self.main_splitter.addWidget(self.viewers_container)
        
        # Create histogram container
        self.histogram_container = QWidget()
        self.histogram_layout = QHBoxLayout(self.histogram_container)
        self.histogram_layout.setContentsMargins(0, 0, 0, 0)
        self.histogram_layout.setSpacing(10)
        
        # Create histogram widgets
        self.original_histogram = HistogramWidget()
        self.original_histogram.set_title("Original Histogram")
        
        self.enhanced_histogram = HistogramWidget()
        self.enhanced_histogram.set_title("Enhanced Histogram")
        
        # Add histograms to layout
        self.histogram_layout.addWidget(self.original_histogram)
        self.histogram_layout.addWidget(self.enhanced_histogram)
        
        # Add histogram container to splitter
        self.main_splitter.addWidget(self.histogram_container)
        
        # Set initial splitter sizes (70% viewers, 30% histograms)
        self.main_splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        layout.addWidget(self.main_splitter)
        
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
    
    def _on_mode_changed(self, index):
        """
        Handle changes to the comparison mode.
        
        Args:
            index: Index of the selected mode
        """
        # Get the selected mode text
        mode_text = self.mode_combo.itemText(index)
        
        # Map mode text to comparison mode
        mode_map = {
            "Side by Side": ComparisonMode.SIDE_BY_SIDE,
            "Overlay": ComparisonMode.OVERLAY, 
            "Split": ComparisonMode.SPLIT
        }
        
        # Set the comparison mode
        self.comparison_mode = mode_map.get(mode_text, ComparisonMode.SIDE_BY_SIDE)
        
        # Update the comparison display
        self._update_comparison()
        
        # Update UI controls based on selected mode
        self._update_ui_for_mode()
    
    def _update_ui_for_mode(self):
        """Update UI controls based on the current comparison mode."""
        # Show/hide viewers based on mode
        if self.comparison_mode == ComparisonMode.SIDE_BY_SIDE:
            # Clear the comparison layout
            for i in reversed(range(self.viewers_layout.count())): 
                item = self.viewers_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            
            # Add original and enhanced viewers
            self.viewers_layout.addWidget(self.original_frame)
            self.viewers_layout.addWidget(self.enhanced_frame)
            
            # Hide comparison frame
            self.comparison_frame.setParent(None)
            
            # Enable/disable sliders
            self.opacity_label.setEnabled(False)
            self.opacity_slider.setEnabled(False)
            self.split_label.setEnabled(False)
            self.split_slider.setEnabled(False)
        else:
            # Clear the comparison layout
            for i in reversed(range(self.viewers_layout.count())):
                item = self.viewers_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            
            # Add comparison viewer
            self.viewers_layout.addWidget(self.comparison_frame)
            
            # Enable/disable sliders based on mode
            self.opacity_label.setEnabled(self.comparison_mode == ComparisonMode.OVERLAY)
            self.opacity_slider.setEnabled(self.comparison_mode == ComparisonMode.OVERLAY)
            self.split_label.setEnabled(self.comparison_mode == ComparisonMode.SPLIT)
            self.split_slider.setEnabled(self.comparison_mode == ComparisonMode.SPLIT)
    
    def _on_opacity_changed(self, value):
        """
        Handle changes to the overlay opacity.
        
        Args:
            value: New opacity value (0-100)
        """
        self.overlay_opacity = value / 100.0
        if self.comparison_mode == ComparisonMode.OVERLAY:
            self._update_comparison()
    
    def _on_split_changed(self, value):
        """
        Handle changes to the split position.
        
        Args:
            value: New split position (0-100)
        """
        self.split_position = value / 100.0
        if self.comparison_mode == ComparisonMode.SPLIT:
            self._update_comparison()
    
    def _on_export_clicked(self):
        """Handle export button click."""
        # Get a save file name from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Comparison", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Create a comparison image based on current mode
            mode = self.comparison_mode.value
            comparison = Exporter.create_comparison_image(
                self.original_image, 
                self.enhanced_image,
                output_path=file_path, 
                mode=mode
            )
            
            # Show status message if parent has a status bar
            parent = self.parent()
            while parent:
                if hasattr(parent, 'status_bar'):
                    parent.status_bar.showMessage(f"Comparison exported to {file_path}")
                    break
                parent = parent.parent()
            
        except Exception as e:
            logger.error(f"Error exporting comparison: {e}")
    
    def _on_histogram_toggled(self, state):
        """Handle histogram visibility toggling."""
        self.show_histograms = state == Qt.CheckState.Checked.value
        self.histogram_container.setVisible(self.show_histograms)
        
        # Adjust splitter sizes
        if self.show_histograms:
            # Show histograms: 70% viewers, 30% histograms
            self.main_splitter.setSizes([700, 300])
        else:
            # Hide histograms: 100% viewers, 0% histograms
            self.main_splitter.setSizes([1, 0])
    
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
        
        # Update histograms
        self.original_histogram.set_image(original)
        self.enhanced_histogram.set_image(enhanced)
        
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