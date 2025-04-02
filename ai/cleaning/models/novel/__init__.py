"""
Novel AI models for medical image enhancement.
These models represent innovative approaches pushing the boundaries of what's possible.
"""

# Import models so they register themselves
from . import diffusion_denoiser as _diffusion_denoiser
from . import swinir_super_resolution as _swinir_sr
from . import stylegan_artifact_removal as _stylegan_ar

__all__ = [
    'diffusion_denoiser',
    'swinir_super_resolution',
    'stylegan_artifact_removal'
]

# Register the models with the expected names
from ai.model_registry import ModelRegistry

ModelRegistry.register("novel_diffusion_denoiser", _diffusion_denoiser.DiffusionDenoiser)
ModelRegistry.register("novel_restormer", _swinir_sr.SwinIRSuperResolution)
ModelRegistry.register("novel_stylegan_artifact_removal", _stylegan_ar.StyleGANArtifactRemoval)