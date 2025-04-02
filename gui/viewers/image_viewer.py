# File: gui/viewers/image_viewer.py

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
    QVBoxLayout, QHBoxLayout, QLabel, QSlider, QToolBar, 
    QGraphicsRectItem, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsTextItem, QPushButton, QColorDialog
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QAction, QIcon, 
    QBrush, QPainterPath, QFont, QTransform
)
from PyQt6.QtCore import (
    Qt, QRectF, QPointF, pyqtSignal, QSize, QEvent, QLine, 
    QObject, QTimer, QLineF
)

from PyQt6.QtWidgets import QGraphicsTextItem
from data.processing.transforms import resize_image
from data.processing.visualization import draw_info_overlay

logger = logging.getLogger(__name__)

class AnnotationMode:
    """Enumeration of annotation modes."""
    NONE = 0
    PAN = 1
    RECTANGLE = 2
    ELLIPSE = 3
    LINE = 4
    ARROW = 5
    TEXT = 6
    MEASURE = 7

class AnnotationItem:
    """Base class for annotation items."""
    def __init__(self, item, mode, data=None):
        self.item = item  # QGraphicsItem
        self.mode = mode  # AnnotationMode
        self.data = data or {}  # Additional data

class ImageViewer(QWidget):
    """
    Widget for displaying and interacting with medical images.
    
    Features:
    - Display of medical images (both standard formats and DICOM)
    - Zoom and pan functionality
    - Region of interest selection
    - Window/level adjustment for medical images
    - Annotations (rectangle, ellipse, line, arrow, text)
    - Measurement tools
    """
    
    # Signals
    imageChanged = pyqtSignal(np.ndarray)  # Emitted when image changes
    regionSelected = pyqtSignal(QRectF)    # Emitted when user selects a region
    annotationAdded = pyqtSignal(object)   # Emitted when an annotation is added
    
    
    
    
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
    
    
    
    
    
    
    def __init__(self, parent=None):
        """Initialize the image viewer."""
        super().__init__(parent)
        self.image_data = None
        self.metadata = {}
        self.zoom_factor = 1.0
        self.window_width = None
        self.window_center = None
        self.annotation_mode = AnnotationMode.NONE
        self.annotations = []
        self.current_annotation_item = None
        self.start_point = None
        self.current_point = None
        self.annotation_pen = QPen(QColor(255, 0, 0))  # Red by default
        self.annotation_pen.setWidth(2)
        self.show_annotations = True
        
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
        self.graphics_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)

        
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
        
        # Add position tracker
        self.position_label = QLabel("Position: ---, ---")
        info_layout.addWidget(self.position_label)
        
        # Add pixel value display
        self.value_label = QLabel("Value: ---")
        info_layout.addWidget(self.value_label)
        
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
        
        self.toolbar.addSeparator()
        
        # Add pan tool
        self.pan_action = QAction("Pan", self)
        self.pan_action.setCheckable(True)
        self.pan_action.triggered.connect(lambda checked: self.set_annotation_mode(
            AnnotationMode.PAN if checked else AnnotationMode.NONE
        ))
        self.toolbar.addAction(self.pan_action)
        
        # Add annotation tools
        self.rect_action = QAction("Rectangle", self)
        self.rect_action.setCheckable(True)
        self.rect_action.triggered.connect(lambda checked: self.set_annotation_mode(
            AnnotationMode.RECTANGLE if checked else AnnotationMode.NONE
        ))
        self.toolbar.addAction(self.rect_action)
        
        self.ellipse_action = QAction("Ellipse", self)
        self.ellipse_action.setCheckable(True)
        self.ellipse_action.triggered.connect(lambda checked: self.set_annotation_mode(
            AnnotationMode.ELLIPSE if checked else AnnotationMode.NONE
        ))
        self.toolbar.addAction(self.ellipse_action)
        
        self.line_action = QAction("Line", self)
        self.line_action.setCheckable(True)
        self.line_action.triggered.connect(lambda checked: self.set_annotation_mode(
            AnnotationMode.LINE if checked else AnnotationMode.NONE
        ))
        self.toolbar.addAction(self.line_action)
        
        self.measure_action = QAction("Measure", self)
        self.measure_action.setCheckable(True)
        self.measure_action.triggered.connect(lambda checked: self.set_annotation_mode(
            AnnotationMode.MEASURE if checked else AnnotationMode.NONE
        ))
        self.toolbar.addAction(self.measure_action)
        
        self.text_action = QAction("Text", self)
        self.text_action.setCheckable(True)
        self.text_action.triggered.connect(lambda checked: self.set_annotation_mode(
            AnnotationMode.TEXT if checked else AnnotationMode.NONE
        ))
        self.toolbar.addAction(self.text_action)
        
        self.toolbar.addSeparator()
        
        # Toggle annotations visibility
        self.show_annotations_action = QAction("Show Annotations", self)
        self.show_annotations_action.setCheckable(True)
        self.show_annotations_action.setChecked(True)
        self.show_annotations_action.triggered.connect(self.toggle_annotations)
        self.toolbar.addAction(self.show_annotations_action)
        
        # Clear annotations
        self.clear_annotations_action = QAction("Clear Annotations", self)
        self.clear_annotations_action.triggered.connect(self.clear_annotations)
        self.toolbar.addAction(self.clear_annotations_action)
        
        # Add toolbar to layout
        layout.addWidget(self.toolbar)
        
        # Set up event handling
        self.graphics_view.viewport().installEventFilter(self)
        
        # Create a timer for tracking mouse movement
        self.tracking_timer = QTimer(self)
        self.tracking_timer.timeout.connect(self.update_pixel_info)
        self.tracking_timer.start(100)  # Update every 100 ms
    
    def eventFilter(self, obj, event):
        if obj is self.graphics_view.viewport():
            if event.type() == QEvent.Type.MouseMove:
                self.handle_mouse_move(event)
                # Add this line to update ongoing annotations
                self.update_current_annotation()
            elif event.type() == QEvent.Type.MouseButtonPress:
                self.handle_mouse_press(event)
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self.handle_mouse_release(event)
        
        return super().eventFilter(obj, event)
    
    def handle_mouse_move(self, event):
        """Handle mouse move events."""
        # Update mouse position in scene coordinates
        scene_pos = self.graphics_view.mapToScene(event.position().toPoint())
        self.current_point = scene_pos
        
        # Update position label
        self.position_label.setText(f"Position: {int(scene_pos.x())}, {int(scene_pos.y())}")
        
        # Handle based on current mode
        if self.annotation_mode == AnnotationMode.PAN and self.start_point is not None:
            # Pan the view
            delta = scene_pos - self.start_point
            self.graphics_view.horizontalScrollBar().setValue(
                int(self.graphics_view.horizontalScrollBar().value() - delta.x())
            )
            self.graphics_view.verticalScrollBar().setValue(
                int(self.graphics_view.verticalScrollBar().value() - delta.y())
            )
    
    def handle_mouse_press(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Get scene position
            scene_pos = self.graphics_view.mapToScene(event.position().toPoint())
            self.start_point = scene_pos
            
            # Handle based on current mode
            if self.annotation_mode == AnnotationMode.PAN:
                self.graphics_view.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            elif self.annotation_mode == AnnotationMode.RECTANGLE:
                # Create a new rectangle
                rect_item = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
                rect_item.setPen(self.annotation_pen)
                self.scene.addItem(rect_item)
                self.current_annotation_item = AnnotationItem(rect_item, AnnotationMode.RECTANGLE)
            elif self.annotation_mode == AnnotationMode.ELLIPSE:
                # Create a new ellipse
                ellipse_item = QGraphicsEllipseItem(QRectF(scene_pos, scene_pos))
                ellipse_item.setPen(self.annotation_pen)
                self.scene.addItem(ellipse_item)
                self.current_annotation_item = AnnotationItem(ellipse_item, AnnotationMode.ELLIPSE)
                
                
                
                
                
                
                
            elif self.annotation_mode == AnnotationMode.LINE or self.annotation_mode == AnnotationMode.MEASURE:
                line_item = QGraphicsLineItem(QLineF(
                    scene_pos.x(), scene_pos.y(),
                    scene_pos.x(), scene_pos.y()
                ))
                line_item.setPen(self.annotation_pen)
                self.scene.addItem(line_item)

                # For measurement, add a text item
                if self.annotation_mode == AnnotationMode.MEASURE:
                    text_item = QGraphicsTextItem("0 px")
                    text_item.setPos(scene_pos + QPointF(5, 5))
                    text_item.setDefaultTextColor(QColor(255, 0, 0))
                    self.scene.addItem(text_item)
                    self.current_annotation_item = AnnotationItem(
                        line_item, AnnotationMode.MEASURE, {"text_item": text_item}
                    )
                else:
                    self.current_annotation_item = AnnotationItem(line_item, AnnotationMode.LINE)
                    
                    
                    
                    
                    
                    
                    
                    
            elif self.annotation_mode == AnnotationMode.TEXT:
                # Create a text item
                text = "Text"  # Default text, could be replaced with a dialog
                text_item = QGraphicsTextItem(text)
                text_item.setPos(scene_pos)
                text_item.setDefaultTextColor(QColor(255, 0, 0))
                self.scene.addItem(text_item)
                text_item.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
                text_item.setFocus()  # Give it focus for immediate editing
                self.current_annotation_item = AnnotationItem(text_item, AnnotationMode.TEXT)
                self.annotations.append(self.current_annotation_item)
                self.current_annotation_item = None
                self.annotationAdded.emit(self.annotations[-1])
    
    
    
    
    
    
    def handle_mouse_release(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Reset cursor if in pan mode
            if self.annotation_mode == AnnotationMode.PAN:
                self.graphics_view.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            
            # Finalize current annotation if any
            if self.current_annotation_item is not None:
                self.annotations.append(self.current_annotation_item)
                self.annotationAdded.emit(self.current_annotation_item)
                self.current_annotation_item = None
            
            self.start_point = None
    
    def update_current_annotation(self):
        """Update the current annotation based on mouse movement."""
        if self.current_annotation_item is None or self.start_point is None:
            return
        
        if self.current_annotation_item.mode == AnnotationMode.RECTANGLE:
            # Update rectangle size
            rect = QRectF(
                min(self.start_point.x(), self.current_point.x()),
                min(self.start_point.y(), self.current_point.y()),
                abs(self.current_point.x() - self.start_point.x()),
                abs(self.current_point.y() - self.start_point.y())
            )
            self.current_annotation_item.item.setRect(rect)
        
        elif self.current_annotation_item.mode == AnnotationMode.ELLIPSE:
            # Update ellipse size
            rect = QRectF(
                min(self.start_point.x(), self.current_point.x()),
                min(self.start_point.y(), self.current_point.y()),
                abs(self.current_point.x() - self.start_point.x()),
                abs(self.current_point.y() - self.start_point.y())
            )
            self.current_annotation_item.item.setRect(rect)
        
        elif self.current_annotation_item.mode == AnnotationMode.LINE or self.current_annotation_item.mode == AnnotationMode.MEASURE:
            # Update line endpoints
            self.current_annotation_item.item.setLine(
                self.start_point.x(), 
                self.start_point.y(),
                self.current_point.x(), 
                self.current_point.y()
    )
            
            # Update measurement text if in measure mode
            if self.current_annotation_item.mode == AnnotationMode.MEASURE and "text_item" in self.current_annotation_item.data:
                # Calculate length
                length = ((self.current_point.x() - self.start_point.x()) ** 2 + 
                         (self.current_point.y() - self.start_point.y()) ** 2) ** 0.5
                
                # Update text
                self.current_annotation_item.data["text_item"].setPlainText(f"{length:.1f} px")
                
                # Position text in the middle of the line
                mid_point = QPointF(
                    (self.start_point.x() + self.current_point.x()) / 2,
                    (self.start_point.y() + self.current_point.y()) / 2
                )
                self.current_annotation_item.data["text_item"].setPos(mid_point + QPointF(5, 5))
    
    def set_annotation_mode(self, mode):
        """Set the current annotation mode."""
        # Reset other action states
        tool_actions = [
            self.pan_action, self.rect_action, self.ellipse_action,
            self.line_action, self.measure_action, self.text_action
        ]
        
        for action in tool_actions:
            action.setChecked(False)
        
        # Set the new mode
        self.annotation_mode = mode
        
        # Update cursor
        if mode == AnnotationMode.NONE:
            self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        elif mode == AnnotationMode.PAN:
            self.graphics_view.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            self.pan_action.setChecked(True)
        elif mode == AnnotationMode.RECTANGLE:
            self.graphics_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.rect_action.setChecked(True)
        elif mode == AnnotationMode.ELLIPSE:
            self.graphics_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.ellipse_action.setChecked(True)
        elif mode == AnnotationMode.LINE:
            self.graphics_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.line_action.setChecked(True)
        elif mode == AnnotationMode.MEASURE:
            self.graphics_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.measure_action.setChecked(True)
        elif mode == AnnotationMode.TEXT:
            self.graphics_view.viewport().setCursor(Qt.CursorShape.IBeamCursor)
            self.text_action.setChecked(True)
    
    def toggle_annotations(self, show):
        """Toggle visibility of annotations."""
        self.show_annotations = show
        for annotation in self.annotations:
            annotation.item.setVisible(show)
            # Also toggle any associated items (like measurement text)
            for key, item in annotation.data.items():
                if hasattr(item, 'setVisible'):
                    item.setVisible(show)
    
    def clear_annotations(self):
        """Clear all annotations."""
        for annotation in self.annotations:
            self.scene.removeItem(annotation.item)
            # Remove any associated items (like measurement text)
            for key, item in annotation.data.items():
                if hasattr(item, 'setVisible'):
                    self.scene.removeItem(item)
        
        self.annotations = []
    
    def update_pixel_info(self):
        """Update pixel value information for current mouse position."""
        if self.image_data is None or self.current_point is None:
            return
        
        # Get pixel coordinates (ensure they're within bounds)
        x, y = int(self.current_point.x()), int(self.current_point.y())
        if x < 0 or y < 0 or x >= self.image_data.shape[1] or y >= self.image_data.shape[0]:
            self.value_label.setText("Value: ---")
            return
        
        # Get pixel value
        if len(self.image_data.shape) == 2:
            # Grayscale
            value = self.image_data[y, x]
            self.value_label.setText(f"Value: {value:.2f}")
        elif len(self.image_data.shape) == 3:
            # RGB or RGBA
            values = self.image_data[y, x]
            if len(values) == 3:
                r, g, b = values
                self.value_label.setText(f"RGB: {r:.2f}, {g:.2f}, {b:.2f}")
            elif len(values) == 4:
                r, g, b, a = values
                self.value_label.setText(f"RGBA: {r:.2f}, {g:.2f}, {b:.2f}, {a:.2f}")
    
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
        
        # Clear annotations for new image
        self.clear_annotations()
    
    def _update_pixmap(self):
        """Update the displayed pixmap from the current image data."""
        if self.image_data is None:
            return
        
        # Create display image (handle windowing if needed)
        if self.window_width is not None and self.window_center is not None:
            from data.io.dicom_handler import DicomHandler
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