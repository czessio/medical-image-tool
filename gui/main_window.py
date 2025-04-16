"""
Main window for medical image enhancement application.
Provides the central hub for all application functionality with a professional medical design.
"""
import os
import logging
import sys
import pickle
import datetime
from pathlib import Path
import numpy as np
from PyQt6.QtGui import QAction, QIcon, QPixmap, QKeySequence, QImage, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QToolBar, QStatusBar, QFileDialog, QMessageBox, QLabel,
    QMenu, QApplication, QDockWidget, QProgressBar, QFrame,
    QGraphicsDropShadowEffect, QPushButton
)
from PyQt6.QtCore import (
    Qt, QSize, QSettings, QThread, QTimer, QMimeData, QUrl,
    pyqtSignal, pyqtSlot
)

from data.io.image_loader import ImageLoader
from data.io.export import Exporter
from ai.cleaning.inference.cleaning_pipeline import CleaningPipeline
from utils.config import Config
from utils.model_service import ModelService

from .dialogs.preferences_dialog import PreferencesDialog
from .viewers.image_viewer import ImageViewer
from .viewers.comparison_view import ComparisonView
from .panels.cleaning_panel import CleaningPanel
from .viewers.dicom_metadata_viewer import DicomMetadataViewer

logger = logging.getLogger(__name__)

class ProcessingThread(QThread):
    """Thread for running image processing operations."""
    finished = pyqtSignal(np.ndarray, dict)  # Signal emitted with processed image and metadata
    progress = pyqtSignal(int)  # Signal for progress updates (0-100)
    error = pyqtSignal(str)  # Signal emitted on error
    
    def __init__(self, pipeline, image, metadata, options):
        """Initialize processing thread."""
        super().__init__()
        self.pipeline = pipeline
        self.image = image
        self.metadata = metadata
        self.options = options
    
    def run(self):
        """Run the processing operation."""
        try:
            # Update the pipeline with options
            # Configure pipeline with selected models
            self.pipeline.disable_all_models()
            
            # Set up denoising model
            if self.options["denoising"]["enabled"]:
                model_id = self.options["denoising"]["model_id"]
                if model_id.startswith("novel_"):
                    if model_id in ["novel_vit_mae_cxr", "novel_resnet50_rad", 
                                   "novel_resnet50_medical", "novel_swinvit"]:
                        self.pipeline._set_enhancement_model(model_id, task_type='enhancement')
                    else:
                        self.pipeline.set_denoising_model(model_id)
                else:
                    self.pipeline.set_denoising_model(model_id)
            
            # Set up super-resolution model
            if self.options["super_resolution"]["enabled"]:
                model_id = self.options["super_resolution"]["model_id"]
                scale_factor = self.options["super_resolution"]["scale_factor"]
                self.pipeline.set_super_resolution_model(model_id, scale_factor=scale_factor)
            
            # Set up artifact removal model
            if self.options["artifact_removal"]["enabled"]:
                model_id = self.options["artifact_removal"]["model_id"]
                self.pipeline.set_artifact_removal_model(model_id)
            
            self.progress.emit(50)  # Models configured
            
            # Process the image
            result = self.pipeline.process(self.image)
            
            self.progress.emit(90)  # Processing complete
            
            # Validate result
            if result is None:
                raise ValueError("Processing returned None")
                
            if not isinstance(result, np.ndarray):
                raise TypeError(f"Result must be a numpy array, got {type(result)}")
                
            if not np.isfinite(result).all():
                # Fix non-finite values
                logger.warning("Result contains non-finite values, replacing with zeros")
                result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Make sure result has the same shape as input
            if result.shape != self.image.shape:
                logger.warning(f"Result shape {result.shape} doesn't match input {self.image.shape}, resizing")
                from data.processing.transforms import resize_image
                result = resize_image(result, (self.image.shape[1], self.image.shape[0]), preserve_aspect_ratio=False)
                
                # Ensure same number of channels
                if len(result.shape) != len(self.image.shape):
                    if len(self.image.shape) == 2 and len(result.shape) == 3:
                        # Convert RGB to grayscale
                        result = np.mean(result, axis=2, keepdims=False)
                    elif len(self.image.shape) == 3 and len(result.shape) == 2:
                        # Convert grayscale to RGB
                        result = np.stack([result] * self.image.shape[2], axis=2)
            
            # Ensure values are in valid range
            if np.issubdtype(result.dtype, np.floating):
                result = np.clip(result, 0.0, 1.0)
            else:
                max_val = 255 if result.dtype == np.uint8 else np.iinfo(result.dtype).max
                result = np.clip(result, 0, max_val)
            
            # Emit the result
            self.finished.emit(result, self.metadata)
            self.progress.emit(100)  # Done
            
        except Exception as e:
            import traceback
            logger.error(f"Error in processing thread: {e}")
            logger.error(traceback.format_exc())
            self.error.emit(str(e))
            self.progress.emit(0)  # Reset progress


class MainWindow(QMainWindow):
    """Main application window for medical image enhancement with a professional design."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        # Initialize data
        self.original_image = None
        self.original_metadata = None
        self.enhanced_image = None
        self.enhanced_metadata = None
        self.is_medical_format = False
        self.last_directory = None
        
        # Create config
        self.config = Config()
        
        # Create model service
        self.model_service = ModelService(self.config)
        
        # Create processing pipeline
        self.pipeline = CleaningPipeline(use_novel_models=self.config.get("models.use_novel", True))
        
        # Create processing thread
        self.processing_thread = None
        
        # Initialize UI
        self.init_ui()
        
        # Load settings
        self.load_settings()
        
        # Set window title
        self.setWindowTitle("Medical Image Enhancement")
        
        # Accept drag and drop
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        """Handle drag enter events for drag-and-drop functionality."""
        if event.mimeData().hasUrls():
            # Check if any of the dragged items are supported image formats
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if any(file_path.lower().endswith(ext) for ext in ImageLoader.get_supported_formats()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Handle drop events for drag-and-drop functionality."""
        if event.mimeData().hasUrls():
            # Get the first image file among dropped items
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if any(file_path.lower().endswith(ext) for ext in ImageLoader.get_supported_formats()):
                    self.load_image_file(file_path)
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Set size
        self.resize(1280, 800)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for main content
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create left panel for cleaning options with shadow effect
        self.left_panel = QWidget()
        left_panel_layout = QVBoxLayout(self.left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create cleaning panel
        self.cleaning_panel = CleaningPanel()
        self.cleaning_panel.cleaningRequested.connect(self.process_image)
        self.cleaning_panel.cleaningRequested.connect(self.process_roi)
        self.cleaning_panel.clearRoiRequested.connect(self.clear_roi)
        
        # Add shadow effect to cleaning panel
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 0)
        self.cleaning_panel.setGraphicsEffect(shadow)
        
        # Add to left panel
        left_panel_layout.addWidget(self.cleaning_panel)
        
        # Create comparison view
        self.comparison_view = ComparisonView()
        
        # Connect ROI selection signals
        self.comparison_view.original_viewer.roi_selected.connect(self.on_roi_selected)
        self.comparison_view.enhanced_viewer.roi_selected.connect(self.on_roi_selected)
        
        # Add drop shadow to comparison view
        shadow2 = QGraphicsDropShadowEffect()
        shadow2.setBlurRadius(15)
        shadow2.setColor(QColor(0, 0, 0, 50))
        shadow2.setOffset(0, 0)
        self.comparison_view.setGraphicsEffect(shadow2)
        
        # Add widgets to splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.comparison_view)
        
        # Set splitter sizes (30% left panel, 70% comparison view)
        self.main_splitter.setSizes([300, 700])
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create progress bar for status bar (initially hidden)
        self.progress_container = QWidget()
        progress_layout = QHBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_label = QLabel("Processing: ")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.progress_bar)
        
        self.status_bar.addPermanentWidget(self.progress_container)
        self.progress_container.hide()
        
        # Add active models indicator to status bar
        self.models_label = QLabel()
        self.models_label.setToolTip("Currently active models")
        self.status_bar.addPermanentWidget(self.models_label)
        self._update_models_label()
        
        # Create toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        
        # Create metadata dock widget
        self.metadata_dock = QDockWidget("DICOM Metadata", self)
        self.metadata_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | 
            Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.metadata_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable | 
            QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        # Create and add the metadata viewer
        self.metadata_viewer = DicomMetadataViewer()
        self.metadata_dock.setWidget(self.metadata_viewer)

        # Add dock widget to main window - initially hidden
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.metadata_dock)
        self.metadata_dock.hide()
        
        # Add actions
        self.create_actions()
        self.create_menus()
        
        # Add actions to toolbar
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.save_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.process_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.export_comparison_action)
    
    def create_actions(self):
        """Create actions for menus and toolbar."""
        # File actions
        self.open_action = QAction("Open", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_image)
        
        self.save_session_action = QAction("Save Session", self)
        self.save_session_action.setShortcut(QKeySequence("Ctrl+Alt+S"))
        self.save_session_action.triggered.connect(self.save_session)
        
        self.load_session_action = QAction("Load Session", self)
        self.load_session_action.setShortcut(QKeySequence("Ctrl+Alt+O"))
        self.load_session_action.triggered.connect(self.load_session)
        
        self.save_action = QAction("Save Enhanced", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_enhanced_image)
        
        self.save_as_action = QAction("Save Enhanced As...", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.setEnabled(False)
        self.save_as_action.triggered.connect(lambda: self.save_enhanced_image(True))
        
        self.export_comparison_action = QAction("Export Comparison", self)
        self.export_comparison_action.setEnabled(False)
        self.export_comparison_action.triggered.connect(self.export_comparison)
        
        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)
        
        # Edit actions
        self.preferences_action = QAction("Preferences", self)
        self.preferences_action.triggered.connect(self.show_preferences)
        
        # Process actions
        self.process_action = QAction("Process Image", self)
        self.process_action.setShortcut(QKeySequence("Ctrl+P"))
        self.process_action.setEnabled(False)
        self.process_action.triggered.connect(self.process_current_image)
        
        # Toggle model type directly from toolbar
        self.toggle_model_action = QAction("Toggle Model Type", self)
        self.toggle_model_action.setShortcut(QKeySequence("Ctrl+T"))
        self.toggle_model_action.triggered.connect(self.toggle_model_type)
        
        # View actions
        self.view_metadata_action = QAction("View DICOM Metadata", self)
        self.view_metadata_action.setCheckable(True)
        self.view_metadata_action.triggered.connect(self._toggle_metadata_viewer)
        self.view_metadata_action.setEnabled(False)  # Initially disabled
        
        # Help actions
        self.about_action = QAction("About", self)
        self.about_action.triggered.connect(self.show_about)
        
        self.help_action = QAction("Help", self)
        self.help_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.help_action.triggered.connect(self.show_help)
    
    def create_menus(self):
        """Create menus for the application."""
        # File menu
        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addAction(self.open_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.save_as_action)
        self.file_menu.addAction(self.export_comparison_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_session_action)
        self.file_menu.addAction(self.load_session_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)
        
        # Edit menu
        self.edit_menu = self.menuBar().addMenu("&Edit")
        self.edit_menu.addAction(self.preferences_action)
        
        # Process menu
        self.process_menu = self.menuBar().addMenu("&Process")
        self.process_menu.addAction(self.process_action)
        self.process_menu.addAction(self.toggle_model_action)
        
        # View menu
        self.view_menu = self.menuBar().addMenu("&View")
        self.view_menu.addAction(self.view_metadata_action)
        
        # Help menu
        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_menu.addAction(self.help_action)
        self.help_menu.addSeparator()
        self.help_menu.addAction(self.about_action)
    
    def _toggle_metadata_viewer(self, checked):
        """Toggle visibility of the DICOM metadata viewer."""
        if checked:
            self.metadata_dock.show()
        else:
            self.metadata_dock.hide()
    
    def load_settings(self):
        """Load application settings."""
        settings = QSettings("MedImageEnhancer", "Application")
        
        # Load window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Load window state
        state = settings.value("windowState")
        if state:
            self.restoreState(state)
        
        # Load last directory
        self.last_directory = settings.value("lastDirectory", str(Path.home()))
        
        # Load splitter sizes
        splitter_sizes = settings.value("splitterSizes")
        if splitter_sizes:
            self.main_splitter.setSizes([int(s) for s in splitter_sizes])
    
    def save_settings(self):
        """Save application settings."""
        settings = QSettings("MedImageEnhancer", "Application")
        
        # Save window geometry
        settings.setValue("geometry", self.saveGeometry())
        
        # Save window state
        settings.setValue("windowState", self.saveState())
        
        # Save last directory
        if self.last_directory:
            settings.setValue("lastDirectory", self.last_directory)
        
        # Save splitter sizes
        settings.setValue("splitterSizes", self.main_splitter.sizes())
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.save_settings()
        event.accept()
    
    def _update_models_label(self):
        """Update the active models indicator in the status bar."""
        if not hasattr(self, 'pipeline') or not hasattr(self.pipeline, 'get_active_models'):
            return
            
        active_models = self.pipeline.get_active_models()
        if not active_models:
            self.models_label.setText("No active models")
            return
            
        model_type = "Novel" if self.pipeline.use_novel_models else "Foundational"
        model_count = len(active_models)
        
        self.models_label.setText(f"{model_type} Models: {model_count} active")
    
    def load_image_file(self, file_path):
        """Load an image from the specified file path."""
        try:
            image_data, metadata, is_medical = ImageLoader.load_image(file_path)
            
            # Store image data
            self.original_image = image_data
            self.original_metadata = metadata
            self.is_medical_format = is_medical
            
            # Clear enhanced image
            self.enhanced_image = None
            self.enhanced_metadata = None
            
            # Update comparison view with just the original image
            self.comparison_view.set_images(image_data, image_data, metadata, metadata)
            
            # Update status bar with image info
            shape_str = f"{image_data.shape[0]}x{image_data.shape[1]}"
            if len(image_data.shape) > 2:
                shape_str += f"x{image_data.shape[2]}"
            
            format_type = "Medical" if is_medical else "Standard"
            self.status_bar.showMessage(f"Loaded {format_type} image: {shape_str}")
            
            # Enable processing
            self.process_action.setEnabled(True)
            
            # Update last directory
            self.last_directory = str(Path(file_path).parent)
            
            # Check if the image has DICOM metadata
            if is_medical:
                # Enable metadata viewer
                self.view_metadata_action.setEnabled(True)
                
                # Update metadata viewer
                self.metadata_viewer.set_metadata(metadata)
                
                # Show metadata viewer if it was previously visible
                if self.view_metadata_action.isChecked():
                    self.metadata_dock.show()
            else:
                # Disable metadata viewer for non-DICOM images
                self.view_metadata_action.setEnabled(False)
                self.metadata_dock.hide()
                self.view_metadata_action.setChecked(False)
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            QMessageBox.critical(self, "Error", f"Error saving image:\n{str(e)}")
    
    def export_comparison(self):
        """Export the current comparison view."""
        if self.original_image is None or self.enhanced_image is None:
            QMessageBox.warning(self, "Warning", "No comparison to export.")
            return
            
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Comparison", self.last_directory, 
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        
        if not file_path:
            return
            
        # Export the comparison
        try:
            result = self.comparison_view.export_comparison(file_path, include_histograms=True)
            
            if result:
                self.status_bar.showMessage(f"Exported comparison to {file_path}")
            else:
                QMessageBox.warning(self, "Warning", "Failed to export comparison.")
                
        except Exception as e:
            logger.error(f"Error exporting comparison: {e}")
            QMessageBox.critical(self, "Error", f"Error exporting comparison:\n{str(e)}")
    
    def toggle_model_type(self):
        """Toggle between foundational and novel models."""
        if self.pipeline.toggle_model_type():
            model_type = "Novel" if self.pipeline.use_novel_models else "Foundational"
            
            # Update radio buttons in the cleaning panel
            if hasattr(self.cleaning_panel, 'use_novel_radio') and hasattr(self.cleaning_panel, 'use_foundational_radio'):
                if self.pipeline.use_novel_models:
                    self.cleaning_panel.use_novel_radio.setChecked(True)
                else:
                    self.cleaning_panel.use_foundational_radio.setChecked(True)
                    
                # Refresh model lists to match the new type
                if hasattr(self.cleaning_panel, 'refresh_model_lists'):
                    self.cleaning_panel.refresh_model_lists()
            
            # Show notification
            self.status_bar.showMessage(f"Switched to {model_type} models")
            
            # Update models label
            self._update_models_label()
        else:
            QMessageBox.warning(self, "Warning", "Failed to toggle model type.")
    
    def process_image(self, options):
        """Process the current image with the specified options."""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "No image to process.")
            return
            
        # Check if a processing thread is already running
        if self.processing_thread is not None and self.processing_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Processing already in progress.")
            return
            
        # Show processing indicator
        self.progress_container.show()
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Processing image...")
        
        # Disable process button during processing
        self.process_action.setEnabled(False)
        
        # Create and start processing thread
        self.processing_thread = ProcessingThread(
            self.pipeline, 
            self.original_image, 
            self.original_metadata, 
            options
        )
        
        # Connect signals
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.progress.connect(self.on_processing_progress)
        self.processing_thread.error.connect(self.on_processing_error)
        
        # Start processing
        self.processing_thread.start()
    
    def process_current_image(self):
        """Process the current image with default options."""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "No image to process.")
            return
            
        # Get options from the cleaning panel
        options = self.cleaning_panel.get_options()
        
        # Process the image
        self.process_image(options)
    
    @pyqtSlot(np.ndarray, dict)
    def on_processing_finished(self, result, metadata):
        """Handle processing thread completion."""
        # Store enhanced image
        self.enhanced_image = result
        self.enhanced_metadata = metadata
        
        # Update comparison view
        self.comparison_view.set_images(
            self.original_image, self.enhanced_image,
            self.original_metadata, self.enhanced_metadata
        )
        
        # Hide progress indicator
        self.progress_container.hide()
        self.status_bar.showMessage("Processing complete")
        
        # Enable save actions
        self.save_action.setEnabled(True)
        self.save_as_action.setEnabled(True)
        self.export_comparison_action.setEnabled(True)
        
        # Re-enable process button
        self.process_action.setEnabled(True)
        
        # Clean up thread
        self.processing_thread = None
        
        # Update models label
        self._update_models_label()
    
    @pyqtSlot(int)
    def on_processing_progress(self, progress):
        """Handle processing progress updates."""
        # Update progress bar
        self.progress_bar.setValue(progress)
    
    @pyqtSlot(str)
    def on_processing_error(self, error_message):
        """Handle processing thread errors."""
        # Hide progress indicator
        self.progress_container.hide()
        
        # Show detailed error message
        error_box = QMessageBox(self)
        error_box.setWindowTitle("Processing Error")
        error_box.setIcon(QMessageBox.Icon.Warning)
        
        error_text = f"""
        <h3>An error occurred during image processing</h3>
        <p>{error_message}</p>
        <p>The application will continue running, but the processed image may not show all expected enhancements.</p>
        <p>Possible solutions:</p>
        <ul>
            <li>Try using a different image</li>
            <li>Disable some enhancement modules</li>
            <li>Switch to {'foundational' if self.pipeline.use_novel_models else 'novel'} models</li>
            <li>Check if the AI models are installed correctly</li>
        </ul>
        """
        
        error_box.setText(error_text)
        error_box.setTextFormat(Qt.TextFormat.RichText)
        error_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        error_box.exec()
        
        # Update status bar
        self.status_bar.showMessage("Processing failed: " + error_message.split('\n')[0])
        
        # Re-enable process button
        self.process_action.setEnabled(True)
        
        # Clean up thread
        self.processing_thread = None


    
    def save_session(self):
        """Save the current session (images and settings)."""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded to save in session.")
            return
            
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", self.last_directory, 
            "Session Files (*.meds);;All Files (*)"
        )
        
        if not file_path:
            return
            
        # Add .meds extension if not present
        if not file_path.lower().endswith('.meds'):
            file_path += '.meds'
        
        try:
            # Create session data
            session = {
                'timestamp': datetime.datetime.now().isoformat(),
                'settings': {
                    'use_novel_models': self.pipeline.use_novel_models,
                    'comparison_mode': self.comparison_view.mode_combo.currentIndex()
                }
            }
            
            # Get options from cleaning panel 
            if hasattr(self.cleaning_panel, 'get_options'):
                session['settings']['options'] = self.cleaning_panel.get_options()
            
            # Save using pickle
            with open(file_path, 'wb') as f:
                # Save original image path or the image itself if necessary
                if 'original_path' in self.original_metadata:
                    session['original_image_path'] = self.original_metadata['original_path']
                else:
                    # If no path available, save the actual image data
                    session['original_image'] = self.original_image
                    session['original_metadata'] = self.original_metadata
                
                # Save enhanced image if available
                if self.enhanced_image is not None:
                    session['enhanced_image'] = self.enhanced_image
                    session['enhanced_metadata'] = self.enhanced_metadata
                
                pickle.dump(session, f)
            
            self.status_bar.showMessage(f"Session saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            QMessageBox.critical(self, "Error", f"Error saving session:\n{str(e)}")
    
    def load_session(self):
        """Load a saved session."""
        # Get load path
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", self.last_directory, 
            "Session Files (*.meds);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Load session data
            with open(file_path, 'rb') as f:
                session = pickle.load(f)
            
            # Load original image
            if 'original_image_path' in session:
                # Load from the original path
                image_path = session['original_image_path']
                if os.path.exists(image_path):
                    self.load_image_file(image_path)
                else:
                    QMessageBox.warning(
                        self, "Warning", 
                        f"Original image file not found: {image_path}\nSession will load without the image."
                    )
                    return
            elif 'original_image' in session:
                # Use the saved image data
                self.original_image = session['original_image']
                self.original_metadata = session.get('original_metadata', {})
                
                # Update comparison view
                if self.original_image is not None:
                    enhanced = session.get('enhanced_image', self.original_image)
                    enhanced_metadata = session.get('enhanced_metadata', self.original_metadata)
                    self.enhanced_image = enhanced
                    self.enhanced_metadata = enhanced_metadata
                    self.comparison_view.set_images(
                        self.original_image, enhanced,
                        self.original_metadata, enhanced_metadata
                    )
            
            # Apply settings
            if 'settings' in session:
                settings = session['settings']
                
                # Apply model type
                use_novel = settings.get('use_novel_models', True)
                if self.pipeline.use_novel_models != use_novel:
                    self.pipeline.toggle_model_type()
                
                # Apply options to cleaning panel if available
                if 'options' in settings and hasattr(self.cleaning_panel, '_load_options'):
                    self.cleaning_panel._load_options(settings['options'])
                else:
                    # Apply UI settings using older method
                    if hasattr(self.cleaning_panel, 'use_novel_radio'):
                        # New UI has radio buttons
                        if use_novel:
                            self.cleaning_panel.use_novel_radio.setChecked(True)
                        else:
                            self.cleaning_panel.use_foundational_radio.setChecked(True)
                    elif hasattr(self.cleaning_panel, 'use_novel_models'):
                        # Old UI has checkbox
                        self.cleaning_panel.use_novel_models.setChecked(use_novel)
                
                # Set comparison mode
                comp_mode_idx = settings.get('comparison_mode', 0)
                self.comparison_view.mode_combo.setCurrentIndex(comp_mode_idx)
            
            # Enable actions if an image is loaded
            if self.original_image is not None:
                self.process_action.setEnabled(True)
            
            # Enable save actions if enhanced image is available
            if self.enhanced_image is not None:
                self.save_action.setEnabled(True)
                self.save_as_action.setEnabled(True)
                self.export_comparison_action.setEnabled(True)
            
            self.status_bar.showMessage(f"Session loaded from {file_path}")
            
            # Update models label
            self._update_models_label()
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            QMessageBox.critical(self, "Error", f"Error loading session:\n{str(e)}")
    
    def open_image(self):
        """Open an image file dialog."""
        # Get supported formats
        formats = ImageLoader.get_supported_formats()
        formats_str = " ".join(f"*{f}" for f in formats)
        
        # Create file filter
        file_filter = f"Image Files ({formats_str});;All Files (*)"
        
        # Show file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.last_directory, file_filter
        )
        
        if file_path:
            self.load_image_file(file_path)
    
    def save_enhanced_image(self, save_as=False):
        """Save the enhanced image."""
        if self.enhanced_image is None:
            QMessageBox.warning(self, "Warning", "No enhanced image to save.")
            return
            
        # Determine save path
        if save_as or 'original_path' not in self.original_metadata:
            # Get supported formats
            formats = ImageLoader.get_supported_formats()
            formats_str = " ".join(f"*{f}" for f in formats)
            
            # Create file filter
            file_filter = f"Image Files ({formats_str});;All Files (*)"
            
            # Show file dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Enhanced Image", self.last_directory, file_filter
            )
            
            if not file_path:
                return
        else:
            # Use original path with "_enhanced" suffix
            orig_path = Path(self.original_metadata['original_path'])
            file_path = str(orig_path.with_name(
                f"{orig_path.stem}_enhanced{orig_path.suffix}"
            ))
        
        # Save the image
        try:
            result = ImageLoader.save_image(
                self.enhanced_image, 
                file_path, 
                self.enhanced_metadata,
                self.is_medical_format
            )
            
            if result:
                self.status_bar.showMessage(f"Saved enhanced image to {file_path}")
            else:
                QMessageBox.warning(self, "Warning", "Failed to save enhanced image.")
                
        except Exception as e:
            logger.error(f"Error saving enhanced image: {e}")
            QMessageBox.critical