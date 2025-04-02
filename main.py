"""
Main entry point for the medical image enhancement application.
Initializes logging, creates the main window, and starts the application.
"""
import sys
import os
from pathlib import Path
import logging
import argparse

from PyQt6.QtWidgets import QApplication

from utils.logging_setup import setup_logging
from gui.main_window import MainWindow

sys.path.insert(0, str(Path(__file__).parent))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Medical Image Enhancement Application")
    
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
        "image_path",
        nargs="?",
        help="Path to an image file to open on startup"
    )
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    log_dir = args.log_dir
    logger = setup_logging(log_dir=log_dir, log_level=log_level)
    
    logger.info("Starting Medical Image Enhancement Application")
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Medical Image Enhancement")
    app.setOrganizationName("MedImageEnhancer")
    
    # Create and show main window
    main_window = MainWindow()
    main_window.show()
    
    # Open image if specified
    if args.image_path:
        try:
            main_window.open_image(args.image_path)
        except Exception as e:
            logger.error(f"Error opening image {args.image_path}: {e}")
    
    # Start the application
    exit_code = app.exec()
    
    logger.info(f"Application exited with code {exit_code}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())