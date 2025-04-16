# File: ai/__init__.py

"""
AI module for the medical image enhancement application.
Provides model loading, inference, and processing functionality.
"""
from .base_model import BaseModel
from .torch_model import TorchModel
from .model_registry import ModelRegistry
from .inference_pipeline import InferencePipeline
from .model_adapter import ModelAdapter
from .model_quantization import ModelQuantizer  # Add the new model quantization module
from .segmentation.segmentation_model import SegmentationModel
from .segmentation.unet_segmentation import UNetSegmentationModel

# Import foundational models to ensure they are registered
from .cleaning.models.foundational.dncnn_denoiser import DnCNNDenoiser
from .cleaning.models.foundational.edsr_super_resolution import EDSRSuperResolution
from .cleaning.models.foundational.unet_artifact_removal import UNetArtifactRemoval

# Import novel models to ensure they are registered
# Define variables to hold the class names so they work with __all__ even if import fails
ResNet50Medical = None
SwinViTModel = None
ViTMAECXR = None

try:
    # Try to import novel models if available
    from .cleaning.models.novel.resnet_50_model import ResNet50Medical
    from .cleaning.models.novel.swinvit_model import SwinViTModel
    from .cleaning.models.novel.vit_mae_cxr_model import ViTMAECXR
except (ImportError, ModuleNotFoundError):
    # Novel models may not be available, that's okay
    pass

__all__ = [
    'BaseModel', 'TorchModel', 'ModelRegistry', 'InferencePipeline',
    'ModelAdapter', 'ModelQuantizer',  # Add the new class
    'SegmentationModel', 'UNetSegmentationModel',
    'DnCNNDenoiser', 'EDSRSuperResolution', 'UNetArtifactRemoval',
    # Add novel models to __all__ if they're available
    'ResNet50Medical', 'SwinViTModel', 'ViTMAECXR'
]