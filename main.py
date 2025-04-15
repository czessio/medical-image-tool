"""
Main entry point for the medical image enhancement application.
Initializes logging, creates the main window, and starts the application.
"""
import sys
import os
from pathlib import Path
import logging
import argparse
import threading
import time

from PyQt6.QtWidgets import QApplication, QSplashScreen, QProgressBar, QVBoxLayout, QLabel, QMessageBox, QWidget
from PyQt6.QtGui import QPixmap, QFont, QPainter
from PyQt6.QtCore import Qt, QTimer, QSize, QByteArray
from PyQt6.QtSvg import QSvgRenderer

from utils.logging_setup import setup_logging
from utils.config import Config
from utils.styles import apply_stylesheet, get_medical_logo
from gui.main_window import MainWindow

sys.path.insert(0, str(Path(__file__).parent))

class SplashScreen(QSplashScreen):
    """Enhanced splash screen with progress bar and SVG logo support."""
    
    def __init__(self):
        # Create the widget that will hold our layout
        self.splash_widget = QWidget()
        
        # Create a pixmap for the splash screen
        pixmap = QPixmap(500, 300)
        pixmap.fill(Qt.GlobalColor.white)
        super().__init__(pixmap)
        
        # Set up the layout
        self._setup_ui()
        
        # Render the logo
        self._render_logo()
    
    def _setup_ui(self):
        """Set up the UI elements for the splash screen."""
        # Create layout for splash screen
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Add logo placeholder (will be painted in paintEvent)
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(QSize(200, 120))
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.logo_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Add title
        self.title_label = QLabel("Medical Image Enhancement")
        self.title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: #0078D7;")
        layout.addWidget(self.title_label)
        
        # Add status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #DDDDDD;
                border-radius: 3px;
                text-align: center;
                background-color: #F5F5F5;
            }
            
            QProgressBar::chunk {
                background-color: #0078D7;
                width: 1px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Add version label
        self.version_label = QLabel("Version 1.0.0")
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.version_label.setStyleSheet("color: #666666;")
        layout.addWidget(self.version_label)
        
        # Apply layout to splash screen
        self.setLayout(layout)
    
    def _render_logo(self):
        """Render the SVG logo for the splash screen."""
        svg_content = get_medical_logo()
        renderer = QSvgRenderer(QByteArray(svg_content.encode()))
        
        pixmap = QPixmap(200, 120)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        
        self.logo_label.setPixmap(pixmap)
    
    def set_progress(self, value, status=None):
        """Update progress bar and status text."""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)
        self.repaint()  # Force repaint to update display

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Medical Image Enhancement Application")
    
    parser.add_argument(
        "--use-foundational",
        action="store_true",
        help="Use foundational models instead of novel models"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory for log files (default: ~/.medimage_enhancer/logs)"
    )
    
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="Disable splash screen"
    )
    
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip model availability checking"
    )
    
    parser.add_argument(
        "--theme",
        choices=["medical", "dark", "high_contrast"],
        default="medical",
        help="Set the application theme"
    )
    
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to an image file to open on startup"
    )
    
    return parser.parse_args()








def initialize_models(splash=None):
    """
    Initialize models, checking and downloading if needed.
    
    Args:
        splash: Optional splash screen to update progress
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        # Create model service
        from utils.model_service import ModelService
        model_service = ModelService()
        
        if splash:
            splash.set_progress(20, "Checking model availability...")
        
        # Initialize models
        status = model_service.initialize()
        
        if splash:
            splash.set_progress(70, "Finalizing model initialization...")
        
        # Check if all required models are available
        foundational_status = model_service.check_category_availability("foundational")
        novel_status = model_service.check_category_availability("novel")
        
        foundational_available = any(foundational_status.values())
        novel_available = any(novel_status.values())
        
        if not foundational_available and not novel_available:
            return False, "No models are available. Please install at least one model."
        
        # Generate a detailed report
        report = []
        if not foundational_available:
            report.append("Foundational models are not available")
        if not novel_available:
            report.append("Novel models are not available")
            
        error_message = ". ".join(report) if report else None
        
        # We can continue if at least one model category is available
        return True, error_message
        
    except Exception as e:
        logging.error(f"Error initializing models: {e}")
        return False, str(e)








def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    log_dir = args.log_dir
    logger = setup_logging(log_dir=log_dir, log_level=log_level)
    
    logger.info("Starting Medical Image Enhancement Application")
    
    # Set model type preference if specified
    if args.use_foundational:
        from utils.config import Config
        config = Config()
        config.set("models.use_novel", False)
        config.save()
        logger.info("Set to use foundational models")
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Medical Image Enhancement")
    app.setOrganizationName("MedImageEnhancer")
    
    # Apply stylesheet based on selected theme
    apply_stylesheet(app, args.theme)
    
    splash = None
    if not args.no_splash:
        # Create and show splash screen
        splash = SplashScreen()
        splash.show()
        app.processEvents()
    
    # Initialize models if not skipped
    if not args.skip_model_check:
        if splash:
            # Create a timer to simulate progress while initializing
            progress = 10
            
            def update_progress():
                nonlocal progress
                if progress < 90:
                    progress += 5
                    splash.set_progress(progress, "Initializing models...")
            
            timer = QTimer()
            timer.timeout.connect(update_progress)
            timer.start(200)  # Update every 200ms
        
        # Initialize models
        success, error_message = initialize_models(splash)
        
        if splash and 'timer' in locals():
            timer.stop()
        
        if not success:
            if splash:
                splash.hide()
            QMessageBox.warning(None, "Model Initialization Warning", 
                            f"Some models may be unavailable: {error_message}\n\n"
                            "The application will start but some features may be limited.")
    
    if splash:
        splash.set_progress(95, "Starting application...")
    
    # Create and show main window
    main_window = MainWindow()
    
    if splash:
        # Show main window and close splash screen
        splash.set_progress(100, "Ready")
        main_window.show()
        splash.finish(main_window)
    else:
        main_window.show()
    
    # Open image if specified
    if args.image_path:
        try:
            main_window.load_image_file(args.image_path)
        except Exception as e:
            logger.error(f"Error opening image {args.image_path}: {e}")
    
    # Start the application
    exit_code = app.exec()
    
    logger.info(f"Application exited with code {exit_code}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())