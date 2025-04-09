"""
Foundational AI models for medical image enhancement.
These are classic models that are widely used and well-established in the field.
"""

# Import models so they register themselves
from . import dncnn_denoiser
from . import edsr_super_resolution
from . import unet_artifact_removal

# Ensure the model classes are available
from .dncnn_denoiser import DnCNNDenoiser
from .edsr_super_resolution import EDSRSuperResolution
from .unet_artifact_removal import UNetArtifactRemoval

# Force registration with the model registry
from ai.model_registry import ModelRegistry
ModelRegistry.register("dncnn_denoiser", DnCNNDenoiser)
ModelRegistry.register("edsr_super_resolution", EDSRSuperResolution)
ModelRegistry.register("unet_artifact_removal", UNetArtifactRemoval)

__all__ = [
    'dncnn_denoiser',
    'edsr_super_resolution',
    'unet_artifact_removal',
    'DnCNNDenoiser',
    'EDSRSuperResolution',
    'UNetArtifactRemoval'
]