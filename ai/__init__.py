# File: ai/__init__.py

"""
AI module for the medical image enhancement application.
Provides model loading, inference, and processing functionality.
"""
from .base_model import BaseModel
from .torch_model import TorchModel
from .model_registry import ModelRegistry
from .inference_pipeline import InferencePipeline
from .segmentation.segmentation_model import SegmentationModel
from .segmentation.unet_segmentation import UNetSegmentationModel

# Import foundational models to ensure they are registered
from .cleaning.models.foundational.dncnn_denoiser import DnCNNDenoiser
from .cleaning.models.foundational.edsr_super_resolution import EDSRSuperResolution
from .cleaning.models.foundational.unet_artifact_removal import UNetArtifactRemoval

# Import novel models to ensure they are registered
try:
    from .cleaning.models.novel.diffusion_denoiser import DiffusionDenoiser
    from .cleaning.models.novel.swinir_super_resolution import SwinIRSuperResolution
    from .cleaning.models.novel.stylegan_artifact_removal import StyleGANArtifactRemoval
except ImportError:
    # Novel models may not be available, that's okay
    pass

__all__ = [
    'BaseModel', 'TorchModel', 'ModelRegistry', 'InferencePipeline',
    'SegmentationModel', 'UNetSegmentationModel',
    'DnCNNDenoiser', 'EDSRSuperResolution', 'UNetArtifactRemoval'
]