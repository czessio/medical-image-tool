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
    CONFIG_UPDATES = {
        "models": {
            "use_novel": False,  # Default to foundational models
            "denoising": {
                "foundational": {
                    "dncnn_denoiser": {
                        "enabled": True,
                        "model_path": "weights/foundational/denoising/dncnn_gray_blind.pth"
                    }
                },
                "device": "auto"  # auto, cpu, cuda
            },
            "super_resolution": {
                "foundational": {
                    "edsr_super_resolution": {
                        "enabled": True,
                        "model_path": "weights/foundational/super_resolution/edsr_x2.pt"
                    }
                },
                "device": "auto",
                "scale_factor": 2  # 2x upscaling by default
            },
            "artifact_removal": {
                "foundational": {
                    "unet_artifact_removal": {
                        "enabled": True,
                        "model_path": "weights/foundational/artifact_removal/G_ema_ep_82.pth"
                    }
                },
                "device": "auto"
            }
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
        
        # Save config to ensure all defaults are written
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
        return Config.DEFAULT_CONFIG.copy()
    
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
                
    def get_model_path(self, model_type, model_name, use_novel=None):
        """
        Get the path to a model's weights file.
        
        Args:
            model_type: Type of model (denoising, super_resolution, artifact_removal)
            model_name: Name of the model
            use_novel: Whether to use novel models (None to use config setting)
            
        Returns:
            str: Path to the model weights file
        """
        if use_novel is None:
            use_novel = self.get("models.use_novel", True)
            
        category = "novel" if use_novel else "foundational"
        
        # First check if this exact model name exists in the config
        model_path = self.get(f"models.{model_type}.{category}.{model_name}.model_path")
        
        if model_path:
            return model_path
            
        # If not found, return a default path based on the model name and category
        weights_dir = self.get("paths.model_weights_dir", "weights")
        return f"{weights_dir}/{category}/{model_name}.pth"
    
    def set_model_path(self, model_type, model_name, model_path, use_novel=None):
        """
        Set the path to a model's weights file.
        
        Args:
            model_type: Type of model (denoising, super_resolution, artifact_removal)
            model_name: Name of the model
            model_path: Path to the model weights file
            use_novel: Whether this is a novel model (None to use model's current category)
        """
        if use_novel is None:
            # Check current category
            if self.get(f"models.{model_type}.novel.{model_name}"):
                category = "novel"
            elif self.get(f"models.{model_type}.foundational.{model_name}"):
                category = "foundational"
            else:
                # Default to novel if model doesn't exist yet
                category = "novel" if self.get("models.use_novel", True) else "foundational"
        else:
            category = "novel" if use_novel else "foundational"
        
        # Ensure the model entry exists
        if not self.get(f"models.{model_type}.{category}.{model_name}"):
            self.set(f"models.{model_type}.{category}.{model_name}", {})
        
        # Set the model path
        self.set(f"models.{model_type}.{category}.{model_name}.model_path", model_path)
        
        # Enable the model by default
        self.set(f"models.{model_type}.{category}.{model_name}.enabled", True)