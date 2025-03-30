"""
Configuration management for the medical image enhancement application.
Handles loading/saving settings and provides default configurations.
"""
import os
import json
import logging
from pathlib import Path

class Config:
    """Manages application configuration settings."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "models": {
            "denoising": {
                "enabled": True,
                "model_path": "weights/cleaning/denoising_model.pth",
                "device": "auto"  # auto, cpu, cuda
            },
            "super_resolution": {
                "enabled": True,
                "model_path": "weights/cleaning/super_resolution_model.pth",
                "device": "auto",
                "scale_factor": 2  # 2x upscaling by default
            },
            "artifact_removal": {
                "enabled": True,
                "model_path": "weights/cleaning/artifact_removal_model.pth",
                "device": "auto"
            }
        },
        "paths": {
            "last_open_dir": "",
            "export_dir": "",
            "temp_dir": "temp"
        },
        "gui": {
            "theme": "dark",
            "window_size": [1024, 768],
            "comparison_view": "side_by_side"  # side_by_side, overlay, split
        },
        "processing": {
            "preview_quality": "medium",  # low, medium, high
            "use_threading": True,
            "max_image_dimension": 2048  # Limit size for preview processing
        }
    }
    
    
    
    def __init__(self, config_path=None):
        """Initialize configuration with optional path to config file."""
        self.logger = logging.getLogger(__name__)
        
        # Set default config path if none provided
        if config_path is None:
            self.config_path = Path(os.path.expanduser("~")) / ".medimage_enhancer" / "config.json"
        else:
            self.config_path = Path(config_path)
            
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create config
        self.config = self.load()
        
        self.save()
    
    
    
    def load(self):
        """Load configuration from file or create default if it doesn't exist."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys are present
                    self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
                    self.logger.info(f"Configuration loaded from {self.config_path}")
                    return loaded_config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            
        # If we get here, either the file doesn't exist or there was an error
        self.logger.info("Using default configuration")
        return self.DEFAULT_CONFIG.copy()
    
    def save(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            return False
    
    def get(self, key_path, default=None):
        """
        Get a configuration value using a dot-separated path.
        Example: config.get("models.denoising.enabled")
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path, value):
        """
        Set a configuration value using a dot-separated path.
        Example: config.set("models.denoising.enabled", False)
        """
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the innermost dictionary
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the value
        config_section[keys[-1]] = value
    
    def _merge_configs(self, default, loaded):
        """Recursively merge loaded config with defaults to ensure all keys exist."""
        for key, value in default.items():
            if key not in loaded:
                loaded[key] = value
            elif isinstance(value, dict) and isinstance(loaded[key], dict):
                self._merge_configs(value, loaded[key])