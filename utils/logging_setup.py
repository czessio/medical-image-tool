"""
Logging configuration for the medical image enhancement application.
Sets up logging for different components of the application.
"""
import os
import logging
import logging.handlers
from pathlib import Path
import datetime

def setup_logging(log_dir=None, log_level=logging.INFO, console_level=logging.INFO):
    """
    Configure application-wide logging.
    
    Args:
        log_dir: Directory to store log files. If None, uses ~/.medimage_enhancer/logs
        log_level: Logging level for file output
        console_level: Logging level for console output
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers
    logger.handlers = []
    
    # Set default log directory if none provided
    if log_dir is None:
        log_dir = Path(os.path.expanduser("~")) / ".medimage_enhancer" / "logs"
    else:
        log_dir = Path(log_dir)
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file handler with rotation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"medimage_{timestamp}.log"
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=5
    )
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log startup message
    logger.info(f"Logging initialized, file: {log_file}")
    
    return logger