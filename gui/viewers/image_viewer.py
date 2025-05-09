"""
Image viewer component for medical image enhancement application.
Provides professional image display with zooming, panning, and measurement tools.
"""
import os
import logging
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QHBoxLayout, QLabel, QSlider, QToolBar, 
    QGraphicsRectItem, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsTextItem, QPushButton, QColorDialog, QFrame, QMenu,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QDialogButtonBox
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QAction, QIcon, 
    QBrush, QPainterPath, QFont, QTransform, QGuiApplication
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
    ROI_SELECT = 8  # New mode for ROI selection

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
    zoomChanged = pyqtSignal(float)        # Emitted when zoom level changes
    roi_selected = pyqtSignal(QRectF)      # Emitted when ROI is selected
    
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
        self.roi_selection = None  # Store current ROI selection
        self.roi_rect = None  # QGraphicsRectItem for ROI display
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Graphics view for image display
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setInteractive(True)
        self.graphics_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        
        # Improve appearance by setting the style
        self.graphics_view.setFrameShape(QFrame.Shape.NoFrame)
        self.graphics_view.setStyleSheet("""
            QGraphicsView {
                background-color: #FFFFFF;
                border: none;
            }
        """)
        
        # Scene to hold the image
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        
        # Pixmap item to display the image
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Add graphics view to layout
        layout.addWidget(self.graphics_view)
        
        # Create info bar layout at the bottom
        info_bar = QFrame()
        info_bar.setFrameShape(QFrame.Shape.StyledPanel)
        info_bar.setStyleSheet("""
            QFrame {
                background-color: #F5F5F5;
                border-top: 1px solid #DDDDDD;
            }
        """)
        
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(5, 2, 5, 2)
        info_layout.setSpacing(10)
        
        # Image info label
        self.info_label = QLabel("No image loaded")
        self.info_label.setStyleSheet("font-weight: bold;")
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
        
        # Add info bar to main layout
        layout.addWidget(info_bar)
        
        # Create toolbar for tools
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setVisible(False)  # Hide by default, can be shown with showToolbar()
        
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
        
        # Add ROI selection tool
        self.roi_action = QAction("Select ROI", self)
        self.roi_action.setCheckable(True)
        self.roi_action.triggered.connect(lambda checked: self.set_annotation_mode(
            AnnotationMode.ROI_SELECT if checked else AnnotationMode.NONE
        ))
        self.toolbar.addAction(self.roi_action)
        
        # Add ROI processing button
        self.process_roi_action = QAction("Process ROI", self)
        self.process_roi_action.setEnabled(False)  # Disabled until ROI is selected
        self.process_roi_action.triggered.connect(self._on_process_roi)
        self.toolbar.addAction(self.process_roi_action)
        
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
    
    def showToolbar(self, show=True):
        """Show or hide the toolbar."""
        self.toolbar.setVisible(show)
    
    def eventFilter(self, obj, event):
        """Filter events for the graphics view."""
        if obj is self.graphics_view.viewport():
            if event.type() == QEvent.Type.MouseMove:
                self.handle_mouse_move(event)
                self.update_current_annotation()
            elif event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.handle_mouse_press(event)
                elif event.button() == Qt.MouseButton.RightButton:
                    self.handle_right_click(event)
                    return True  # Consume right-click events
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.handle_mouse_release(event)
            
            # Add wheel event handling
            elif event.type() == QEvent.Type.Wheel:
                self._wheel_event(event)
                return True  # Consume the event
        
        return super().eventFilter(obj, event)
    
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
        
        # Emit signal for zoom change
        self.zoomChanged.emit(self.zoom_factor)
    
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
                    
            elif self.annotation_mode == AnnotationMode.ROI_SELECT:
                # Remove existing ROI rect if any
                if self.roi_rect is not None:
                    self.scene.removeItem(self.roi_rect)
                
                # Create a new ROI rectangle
                self.roi_rect = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
                self.roi_rect.setPen(QPen(QColor(0, 255, 0), 2))  # Green pen for ROI
                self.scene.addItem(self.roi_rect)
                self.current_annotation_item = None  # Don't track as a regular annotation
                                      
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
            
            # Handle ROI selection
            if self.annotation_mode == AnnotationMode.ROI_SELECT and self.roi_rect is not None:
                # Finalize ROI selection
                rect = self.roi_rect.rect()
                
                # Only accept ROI if it has a reasonable size
                if rect.width() > 10 and rect.height() > 10:
                    self.roi_selection = rect
                    self.process_roi_action.setEnabled(True)
                    
                    # Emit signal with ROI
                    self.roi_selected.emit(rect)
                else:
                    # Remove too small ROI
                    self.scene.removeItem(self.roi_rect)
                    self.roi_rect = None
                    self.roi_selection = None
                    self.process_roi_action.setEnabled(False)
            
            self.start_point = None
    
    def handle_right_click(self, event):
        """Handle right-click context menu."""
        if self.image_data is None:
            return
        
        # Create context menu
        menu = QMenu(self)
        
        # Add window/level adjustment if it's a medical image
        if self.metadata and ('WindowCenter' in self.metadata or 'WindowWidth' in self.metadata):
            window_level_action = menu.addAction("Adjust Window/Level")
            window_level_action.triggered.connect(self.show_window_level_dialog)
        
        # Add reset window/level if it's currently being applied
        if self.window_width is not None or self.window_center is not None:
            reset_window_level_action = menu.addAction("Reset Window/Level")
            reset_window_level_action.triggered.connect(self.reset_window_level)
        
        # Add ROI selection option
        roi_action = menu.addAction("Select ROI")
        roi_action.triggered.connect(lambda: self.set_annotation_mode(AnnotationMode.ROI_SELECT))
        
        # Add "Copy image info" option
        copy_info_action = menu.addAction("Copy Image Info")
        copy_info_action.triggered.connect(self.copy_image_info)
        
        # Show the menu at the mouse position
        menu.exec(event.globalPosition().toPoint())
    
    def update_current_annotation(self):
        """Update the current annotation based on mouse movement."""
        if self.current_annotation_item is None and self.start_point is None:
            return
        
        # Special handling for ROI selection
        if self.annotation_mode == AnnotationMode.ROI_SELECT and self.roi_rect is not None and self.start_point is not None:
            # Update rectangle size
            rect = QRectF(
                min(self.start_point.x(), self.current_point.x()),
                min(self.start_point.y(), self.current_point.y()),
                abs(self.current_point.x() - self.start_point.x()),
                abs(self.current_point.y() - self.start_point.y())
            )
            self.roi_rect.setRect(rect)
            return
        
        if self.current_annotation_item is None:
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
                self.current_annotation_item.data["text_item"].setPos(mid_point)
    
    def set_annotation_mode(self, mode):
        """Set the current annotation mode."""
        # Reset other action states
        tool_actions = [
            self.pan_action, self.rect_action, self.ellipse_action,
            self.line_action, self.measure_action, self.text_action,
            self.roi_action  # Add ROI action to the list
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
        elif mode == AnnotationMode.ROI_SELECT:
            self.graphics_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.roi_action.setChecked(True)
            
            # Clear existing ROI if any
            if self.roi_rect is not None:
                self.scene.removeItem(self.roi_rect)
                self.roi_rect = None
                self.roi_selection = None
                self.process_roi_action.setEnabled(False)
    
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
        
        # Clear ROI
        self.clear_roi()
    
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
        
        # Make sure display_image is contiguous in memory
        if not display_image.flags['C_CONTIGUOUS']:
            display_image = np.ascontiguousarray(display_image)
        
        if len(display_image.shape) == 2:
            # Grayscale image
            q_image = QImage(
                display_image.tobytes(),  # Use tobytes() instead of data
                width, height,
                width,  # Bytes per line
                QImage.Format.Format_Grayscale8
            )
        elif display_image.shape[2] == 3:
            # RGB image
            q_image = QImage(
                display_image.tobytes(),  # Use tobytes() instead of data
                width, height,
                width * 3,  # Bytes per line (3 channels)
                QImage.Format.Format_RGB888
            )
        elif display_image.shape[2] == 4:
            # RGBA image
            q_image = QImage(
                display_image.tobytes(),  # Use tobytes() instead of data
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
        """Zoom in on the image."""
        # Increase zoom factor by 20%
        self.zoom_factor *= 1.2
        
        # Apply scaling
        transform = QTransform()
        transform.scale(self.zoom_factor, self.zoom_factor)
        self.graphics_view.setTransform(transform)
        
        # Update zoom label
        self._update_zoom_label()
        
        # Emit signal for zoom change
        self.zoomChanged.emit(self.zoom_factor)
    
    def zoom_out(self):
        """Zoom out from the image."""
        # Decrease zoom factor by 20%
        self.zoom_factor /= 1.2
        
        # Apply scaling
        transform = QTransform()
        transform.scale(self.zoom_factor, self.zoom_factor)
        self.graphics_view.setTransform(transform)
        
        # Update zoom label
        self._update_zoom_label()
        
        # Emit signal for zoom change
        self.zoomChanged.emit(self.zoom_factor)
    
    def zoom_fit(self):
        """Fit the image to the view."""
        if self.image_data is None:
            return
            
        # Reset zoom factor
        self.zoom_factor = 1.0
        
        # Fit scene in view
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
        # Calculate actual zoom factor based on the view scale
        transform = self.graphics_view.transform()
        self.zoom_factor = transform.m11()  # Scaling factor from the transformation matrix
        
        # Update zoom label
        self._update_zoom_label()
        
        # Emit signal for zoom change
        self.zoomChanged.emit(self.zoom_factor)
    
    def _update_zoom_label(self):
        """Update the zoom level display."""
        # Ensure zoom factor is positive
        self.zoom_factor = max(0.01, self.zoom_factor)
        
        # Update label
        self.zoom_label.setText(f"Zoom: {self.zoom_factor * 100:.0f}%")
    
    def clear_roi(self):
        """Clear the current ROI selection."""
        if self.roi_rect is not None:
            self.scene.removeItem(self.roi_rect)
            self.roi_rect = None
            
        self.roi_selection = None
        if hasattr(self, 'process_roi_action'):
            self.process_roi_action.setEnabled(False)
    
    def _on_process_roi(self):
        """Handle the process ROI action."""
        if self.roi_selection is None:
            return
            
        # Convert ROI from scene coordinates to image coordinates
        x = int(self.roi_selection.x())
        y = int(self.roi_selection.y())
        width = int(self.roi_selection.width())
        height = int(self.roi_selection.height())
        
        # Emit signal with rectangular coordinates (x, y, width, height)
        rect = (x, y, width, height)
        if hasattr(self, 'roi_selected'):
            self.roi_selected.emit(rect)