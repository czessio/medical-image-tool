"""
Image viewer component for medical image enhancement application.
Provides basic image display with zooming, panning, and measurement tools.
"""
import os
import logging
from pathlib import Path
import numpy as np



from PyQt6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QHBoxLayout, QLabel, QSlider, QToolBar
)
from PyQt6.QtGui import QAction



from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIcon
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QSize

from data.processing.transforms import resize_image
from data.processing.visualization import draw_info_overlay

logger = logging.getLogger(__name__)

class ImageViewer(QWidget):
    """
    Widget for displaying and interacting with medical images.
    
    Features:
    - Display of medical images (both standard formats and DICOM)
    - Zoom and pan functionality
    - Region of interest selection
    - Window/level adjustment for medical images
    """
    
    # Signals
    imageChanged = pyqtSignal(np.ndarray)  # Emitted when image changes
    regionSelected = pyqtSignal(QRectF)    # Emitted when user selects a region
    
    def __init__(self, parent=None):
        """Initialize the image viewer."""
        super().__init__(parent)
        self.image_data = None
        self.metadata = {}
        self.zoom_factor = 1.0
        self.window_width = None
        self.window_center = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Graphics view for image display
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setInteractive(True)
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        
        # Scene to hold the image
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        
        # Pixmap item to display the image
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Add graphics view to layout
        layout.addWidget(self.graphics_view)
        
        # Create info bar layout at the bottom
        info_layout = QHBoxLayout()
        
        # Image info label
        self.info_label = QLabel("No image loaded")
        info_layout.addWidget(self.info_label)
        
        # Zoom level display
        self.zoom_label = QLabel("Zoom: 100%")
        info_layout.addWidget(self.zoom_label)
        
        # Add info layout to main layout
        layout.addLayout(info_layout)
        
        # Create toolbar for tools
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(24, 24))
        
        # Add zoom in/out actions
        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.triggered.connect(self.zoom_in)
        self.toolbar.addAction(self.zoom_in_action)
        
        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.triggered.connect(self.zoom_out)
        self.toolbar.addAction(self.zoom_out_action)
        
        self.zoom_fit_action = QAction("Fit to View", self)
        self.zoom_fit_action.triggered.connect(self.zoom_fit)
        self.toolbar.addAction(self.zoom_fit_action)
        
        # Add toolbar to layout
        layout.addWidget(self.toolbar)
        
        # Set up wheelEvent for zooming
        self.graphics_view.wheelEvent = self._wheel_event
    
    def set_image(self, image_data, metadata=None):
        """
        Set the image to display.
        
        Args:
            image_data: Numpy array of image data
            metadata: Optional dictionary of image metadata
        """
        if image_data is None:
            return
        
        self.image_data = image_data
        self.metadata = metadata or {}
        
        # Convert the numpy array to QImage
        self._update_pixmap()
        
        # Update info label
        if image_data is not None:
            shape_str = f"{image_data.shape[0]}x{image_data.shape[1]}"
            if len(image_data.shape) > 2:
                shape_str += f"x{image_data.shape[2]}"
            self.info_label.setText(f"Image: {shape_str}")
        else:
            self.info_label.setText("No image loaded")
        
        # Emit signal
        self.imageChanged.emit(image_data)
        
        # Fit to view
        self.zoom_fit()
    
    def _update_pixmap(self):
        """Update the displayed pixmap from the current image data."""
        if self.image_data is None:
            return
        
        # Create display image (handle windowing if needed)
        if self.window_width is not None and self.window_center is not None:
            from ...data.io.dicom_handler import DicomHandler
            display_image = DicomHandler.apply_window_level(
                self.image_data, self.metadata,
                window=self.window_width, 
                level=self.window_center
            )
        else:
            display_image = self.image_data
        
        # Convert the numpy array to QImage
        if np.issubdtype(display_image.dtype, np.floating):
            display_image = (display_image * 255).clip(0, 255).astype(np.uint8)
        
        height, width = display_image.shape[:2]
        
        if len(display_image.shape) == 2:
            # Grayscale image
            q_image = QImage(
                display_image.data,
                width, height,
                width,  # Bytes per line
                QImage.Format.Format_Grayscale8
            )
        elif display_image.shape[2] == 3:
            # RGB image
            q_image = QImage(
                display_image.data,
                width, height,
                width * 3,  # Bytes per line (3 channels)
                QImage.Format.Format_RGB888
            )
        elif display_image.shape[2] == 4:
            # RGBA image
            q_image = QImage(
                display_image.data,
                width, height,
                width * 4,  # Bytes per line (4 channels)
                QImage.Format.Format_RGBA8888
            )
        else:
            logger.error(f"Unsupported image format: {display_image.shape}")
            return
        
        # Create pixmap from QImage
        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)
        
        # Update scene rect
        self.scene.setSceneRect(QRectF(0, 0, width, height))
    
    def zoom_in(self):
        """Zoom in by a fixed factor."""
        self.zoom_factor *= 1.2
        self.graphics_view.scale(1.2, 1.2)
        self._update_zoom_label()
    
    def zoom_out(self):
        """Zoom out by a fixed factor."""
        self.zoom_factor /= 1.2
        self.graphics_view.scale(1/1.2, 1/1.2)
        self._update_zoom_label()
    
    def zoom_fit(self):
        """Zoom to fit the image in the view."""
        if self.pixmap_item.pixmap().isNull():
            return
            
        # Reset the view
        self.graphics_view.resetTransform()
        self.zoom_factor = 1.0
        
        # Calculate the scale to fit
        view_rect = self.graphics_view.viewport().rect()
        scene_rect = self.scene.sceneRect()
        
        x_scale = view_rect.width() / scene_rect.width()
        y_scale = view_rect.height() / scene_rect.height()
        
        # Use the smaller scale to fit entirely in the view
        scale = min(x_scale, y_scale) * 0.95  # 5% margin
        
        # Apply the scale
        self.graphics_view.scale(scale, scale)
        self.zoom_factor = scale
        
        self._update_zoom_label()
    
    def _update_zoom_label(self):
        """Update the zoom level display."""
        percent = int(self.zoom_factor * 100)
        self.zoom_label.setText(f"Zoom: {percent}%")
    
    def _wheel_event(self, event):
        """Handle mouse wheel events for zooming."""
        delta = event.angleDelta().y()
        
        if delta > 0:
            # Zoom in
            factor = 1.1
        else:
            # Zoom out
            factor = 1/1.1
        
        # Record the scene position before zoom
        view_pos = event.position()
        scene_pos = self.graphics_view.mapToScene(int(view_pos.x()), int(view_pos.y()))
        
        # Apply zoom
        self.graphics_view.scale(factor, factor)
        self.zoom_factor *= factor
        
        # Get the new scene position and translate to maintain position under cursor
        new_scene_pos = self.graphics_view.mapToScene(int(view_pos.x()), int(view_pos.y()))
        delta_scene_pos = new_scene_pos - scene_pos
        self.graphics_view.translate(delta_scene_pos.x(), delta_scene_pos.y())
        
        self._update_zoom_label()
    
    def set_window_level(self, window, level):
        """
        Set the window/level (contrast/brightness) for medical images.
        
        Args:
            window: Window width (contrast)
            level: Window center (brightness)
        """
        self.window_width = window
        self.window_center = level
        self._update_pixmap()
    
    def reset_window_level(self):
        """Reset the window/level to default values."""
        self.window_width = None
        self.window_center = None
        self._update_pixmap()
    
    def show_info_overlay(self, show=True):
        """
        Show or hide information overlay on the image.
        
        Args:
            show: Whether to show the overlay
        """
        if self.image_data is None:
            return
        
        if show:
            # Create info text based on metadata
            info_lines = []
            
            # Add basic image info
            height, width = self.image_data.shape[:2]
            info_lines.append(f"Size: {width}x{height}")
            
            # Add DICOM metadata if available
            if 'PatientID' in self.metadata:
                info_lines.append(f"Patient ID: {self.metadata['PatientID']}")
            if 'PatientName' in self.metadata:
                info_lines.append(f"Patient: {self.metadata['PatientName']}")
            if 'Modality' in self.metadata:
                info_lines.append(f"Modality: {self.metadata['Modality']}")
            
            info_text = "\n".join(info_lines)
            
            # Add overlay to image
            display_image = draw_info_overlay(self.image_data, info_text)
            self.set_image(display_image, self.metadata)
        else:
            # Restore original image
            self._update_pixmap()