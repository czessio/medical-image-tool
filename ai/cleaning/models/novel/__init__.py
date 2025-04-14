# Add this line to the imports
from .resnet_50_model import ResNet50Medical
from .swinvit_model import SwinViTModel
from ai.model_registry import ModelRegistry
from .vit_mae_cxr_model import ViTMAECXR
from .resnet_50_rad_model import ResNet50RadImageNet

# Add to the registration section
ModelRegistry.register("novel_vit_mae_cxr", ViTMAECXR)

# Add to the __all__ list
__all__ = [
    'diffusion_denoiser',
    'swinir_super_resolution',
    'stylegan_artifact_removal',
    'swinvit_model',
    'resnet_50_model',
    'resnet_50_rad_model',
    'vit_mae_cxr_model',  # Add the new model module
    'DiffusionDenoiser', 
    'SwinIRSuperResolution',
    'StyleGANArtifactRemoval',
    'SwinViTModel',
    'ResNet50Medical',
    'ResNet50RadImageNet',
    'ViTMAECXR'  # Add the new model class
]