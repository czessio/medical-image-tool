import os
import tempfile
from pathlib import Path
import sys

# Add the parent directory to sys.path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.config import Config

def test_config_creation():
    """Test that default config is created properly"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "test_config.json"
        config = Config(config_path)
        
        # Check default values are set
        assert config.get("models.denoising.enabled") == True
        assert config.get("gui.theme") == "dark"
        
        # Check config file was created
        assert config_path.exists()

def test_config_save_load():
    """Test saving and loading config values"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "test_config.json"
        config = Config(config_path)
        
        # Modify a value
        config.set("models.denoising.enabled", False)
        config.save()
        
        # Create new config instance and load
        config2 = Config(config_path)
        assert config2.get("models.denoising.enabled") == False

def test_config_get_default():
    """Test getting values with defaults"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "test_config.json"
        config = Config(config_path)
        
        # Try to get a non-existent key
        result = config.get("non.existent.key", "default_value")
        assert result == "default_value"