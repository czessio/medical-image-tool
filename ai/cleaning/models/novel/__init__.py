"""
Novel AI models for medical image enhancement.
These models represent innovative approaches pushing the boundaries of what's possible.
"""

# Import models so they register themselves
from . import diffusion_denoiser
from . import swinir_super_resolution
from . import stylegan_artifact_removal

__all__ = [
    'diffusion_denoiser',
    'swinir_super_resolution',
    'stylegan_artifact_removal'
]