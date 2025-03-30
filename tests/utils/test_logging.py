# File: main_2.0/tests/test_logging.py
import os
import logging
import tempfile
from pathlib import Path
import sys

# Add the parent directory to sys.path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_setup import setup_logging

def test_logging_creates_file():
    """Test that log file is created"""
    tmp_dir = tempfile.mkdtemp()  # Use mkdtemp instead of TemporaryDirectory
    try:
        log_dir = Path(tmp_dir)
        logger = setup_logging(log_dir=log_dir)
        
        # Log a test message
        logger.info("Test log message")
        
        # Check that a log file was created
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0
        
        # Remove all handlers to close files
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    finally:
        # Clean up temp files manually
        for file in Path(tmp_dir).glob("*.log"):
            try:
                os.remove(file)
            except:
                pass
        os.rmdir(tmp_dir)

def test_logging_levels():
    """Test that log levels work properly"""
    tmp_dir = tempfile.mkdtemp()  # Use mkdtemp instead of TemporaryDirectory
    try:
        log_dir = Path(tmp_dir)
        logger = setup_logging(log_dir=log_dir, log_level=logging.ERROR)
        
        # These shouldn't be written to file
        logger.debug("Debug message")
        logger.info("Info message")
        
        # This should be written
        logger.error("Error message")
        
        # Remove all handlers to close files
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        # Check log file content
        log_files = list(log_dir.glob("*.log"))
        with open(log_files[0], 'r') as f:
            content = f.read()
            assert "Debug message" not in content
            assert "Info message" not in content
            assert "Error message" in content
    finally:
        # Clean up temp files manually
        for file in Path(tmp_dir).glob("*.log"):
            try:
                os.remove(file)
            except:
                pass
        os.rmdir(tmp_dir)