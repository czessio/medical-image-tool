"""
Novel AI models for medical image enhancement.
These models represent innovative approaches pushing the boundaries of what's possible.
"""

# Import models so they register themselves
try:
    from . import diffusion_denoiser
    from . import swinir_super_resolution
    from . import stylegan_artifact_removal

    # Ensure the model classes are available
    from .diffusion_denoiser import DiffusionDenoiser
    from .swinir_super_resolution import SwinIRSuperResolution
    from .stylegan_artifact_removal import StyleGANArtifactRemoval

    # Register the models with the expected names
    from ai.model_registry import ModelRegistry
    ModelRegistry.register("novel_diffusion_denoiser", DiffusionDenoiser)
    ModelRegistry.register("novel_restormer", SwinIRSuperResolution)
    ModelRegistry.register("novel_stylegan_artifact_removal", StyleGANArtifactRemoval)

    __all__ = [
        'diffusion_denoiser',
        'swinir_super_resolution',
        'stylegan_artifact_removal',
        'DiffusionDenoiser',
        'SwinIRSuperResolution',
        'StyleGANArtifactRemoval'
    ]
except ImportError:
    # Novel models may not be available, that's okay
    __all__ = []