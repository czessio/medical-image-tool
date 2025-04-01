"""
Foundational AI models for medical image enhancement.
These are classic models that are widely used and well-established in the field.
"""

# Import models so they register themselves
from . import dncnn_denoiser
from . import edsr_super_resolution
from . import unet_artifact_removal

__all__ = [
    'dncnn_denoiser',
    'edsr_super_resolution',
    'unet_artifact_removal'
]