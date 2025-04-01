"""
StyleGAN-inspired artifact removal model implementation.
Uses generative adversarial network concepts for medical image artifact removal.
"""
import os
import logging
import numpy as np
from typing import Optional, Tuple, Union, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ....torch_model import TorchModel
from ....model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization module used in StyleGAN.
    """
    def __init__(self, style_dim, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style = nn.Linear(style_dim, channels * 2)
        
    def forward(self, x, s):
        style = self.style(s)
        gamma, beta = style.chunk(2, dim=1)
        
        # Reshape to match feature map dimensions
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        # Apply instance normalization then modulate with style
        out = self.norm(x)
        out = gamma * out + beta
        
        return out

class NoiseInjection(nn.Module):
    """
    Noise injection module used in StyleGAN.
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
        
        
        
    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        else:
            # Ensure noise has compatible dimensions with x
            if noise.shape[2:] != x.shape[2:]:
                noise = F.interpolate(noise, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return x + self.weight * noise




class StyledConvBlock(nn.Module):
    """
    StyleGAN-inspired convolutional block with AdaIN and noise injection.
    """
    def __init__(self, in_channels, out_channels, style_dim, kernel_size=3, upsample=False):
        super().__init__()
        self.upsample = upsample
        
        # Padding size to maintain spatial dimensions
        padding = kernel_size // 2
        
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        
        # Noise injection and AdaIN layers
        self.noise = NoiseInjection(out_channels)
        self.adain = AdaIN(style_dim, out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, style, noise=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
        x = self.conv(x)
        x = self.noise(x, noise)
        x = self.adain(x, style)
        x = self.activation(x)
        
        return x

class MappingNetwork(nn.Module):
    """
    Mapping network to generate style vectors from latent codes.
    """
    def __init__(self, latent_dim=512, style_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(latent_dim if i == 0 else style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
            
        self.mapping = nn.Sequential(*layers)
        self.style_dim = style_dim
        
    def forward(self, z):
        return self.mapping(z)

class StyleEncoder(nn.Module):
    """
    Encoder network to extract styles from input images.
    """
    def __init__(self, in_channels=1, style_dim=512):
        super().__init__()
        
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)  # Downscale
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)  # Downscale
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)  # Downscale
        
        # Global average pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, style_dim)
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        
        x = self.global_pool(x).view(x.shape[0], -1)
        x = self.fc(x)
        
        return x

class AttentionBlock(nn.Module):
    """
    Self-attention block for focusing on important regions.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, channels, height, width = x.shape
        
        q = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, height * width)
        v = self.value(x).view(batch, -1, height * width)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        return self.gamma * out + x

class StyleGANGenerator(nn.Module):
    """
    StyleGAN-inspired generator for artifact removal.
    """
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        style_dim=512, 
        initial_size=4,
        num_layers=6
    ):
        super().__init__()
        self.style_dim = style_dim
        self.initial_size = initial_size
        
        # Input processing
        self.input_conv = nn.Conv2d(in_channels, 64, 3, 1, 1)
        
        # Style encoder and mapping network
        self.style_encoder = StyleEncoder(in_channels, style_dim)
        self.mapping = MappingNetwork(style_dim, style_dim)
        
        # Initial constant tensor (learnable)
        self.constant = nn.Parameter(torch.randn(1, 64, initial_size, initial_size))
        
        # Styled convolution blocks
        self.blocks = nn.ModuleList()
        
        # Channel sizes for each resolution level
        channels = [64, 64, 128, 128, 64, 64, 32]
        
        # Add styled conv blocks with upsampling at appropriate layers
        for i in range(num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            upsample = i % 2 == 0 and i > 0  # Upsample every other layer starting from second
            self.blocks.append(StyledConvBlock(in_ch, out_ch, style_dim, upsample=upsample))
        
        # Attention block at middle layers
        self.attention = AttentionBlock(128)
        
        # Output layer
        self.to_rgb = nn.Conv2d(channels[-1], out_channels, 1, 1, 0)
        
        # U-Net skip connection
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
    def get_latent(self, batch_size, device):
        return torch.randn(batch_size, self.style_dim, device=device)
    
    def forward(self, x, return_latents=False, inject_noise=True):
        # Extract style from input image
        img_style = self.style_encoder(x)
        
        # Process style through mapping network
        style = self.mapping(img_style)
        
        # Skip connection for U-Net like behavior
        skip = self.skip_connection(x)
        
        # Start with constant input
        batch_size = x.shape[0]
        out = self.constant.repeat(batch_size, 1, 1, 1)
        
        # Apply styled conv blocks
        for i, block in enumerate(self.blocks):
            # Generate noise for each block if using noise
            noise = torch.randn(batch_size, 1, out.shape[2], out.shape[3], device=x.device) if inject_noise else None
            
            # Apply block
            out = block(out, style, noise)
            
            # Apply attention at middle layer
            if i == 2:  # Apply attention at middle resolution
                out = self.attention(out)
        
        # Final RGB output
        out = self.to_rgb(out)
        
        # Add skip connection and ensure output is normalized
        out = torch.tanh(out) * 0.5 + 0.5
        
        # Resize output to match input dimensions if needed
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Add skip connection (residual learning)
        out = out + skip
        
        # Normalize to 0-1 range
        out = torch.clamp(out, 0, 1)
        
        if return_latents:
            return out, style
        return out

class Discriminator(nn.Module):
    """
    PatchGAN-style discriminator for artifact removal.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            # input: in_channels x H x W
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x H/2 x W/2
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x H/4 x W/4
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 x H/8 x W/8
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 x H/8-1 x W/8-1
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
            # Output 1-channel PatchGAN feature map
        )
        
    def forward(self, x):
        return self.layers(x)

class StyleGANArtifactRemoval(TorchModel):
    """
    StyleGAN-inspired artifact removal model.
    
    Uses a StyleGAN generator architecture with a perceptual approach
    to remove artifacts from medical images while preserving details.
    """
    
    def __init__(self, model_path=None, device=None, inject_noise=False):
        """
        Initialize the StyleGAN artifact removal model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            inject_noise: Whether to inject noise during generation (can help with diversity)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        self.inject_noise = inject_noise
        super().__init__(model_path, device)
    
    def _create_model_architecture(self):
        """Create the StyleGAN artifact removal model architecture."""
        # For medical images, typically grayscale input/output
        model = StyleGANGenerator(
            in_channels=1,     # Grayscale input
            out_channels=1,    # Grayscale output
            style_dim=512,
            initial_size=8,
            num_layers=6
        )
        
        return model
    
    def preprocess(self, image):
        """
        Preprocess the input image for the StyleGAN model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Use parent preprocessing but ensure channels are correct
        tensor = super().preprocess(image)
        
        # If input is RGB, convert to grayscale (average channels)
        if tensor.shape[1] == 3:
            tensor = tensor.mean(dim=1, keepdim=True)
        
        return tensor
    
    def inference(self, preprocessed_tensor):
        """
        Run inference with the StyleGAN model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Artifact-free output tensor
        """
        with torch.no_grad():
            return self.model(preprocessed_tensor, inject_noise=self.inject_noise)
    
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output.
        
        Args:
            model_output: Raw output from the model
            original_image: Original input image (not used)
            
        Returns:
            numpy.ndarray: Artifact-free image as numpy array
        """
        return super().postprocess(model_output, original_image)

# Register the model
ModelRegistry.register("stylegan_artifact_removal", StyleGANArtifactRemoval)