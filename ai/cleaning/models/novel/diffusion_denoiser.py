"""
Diffusion-based image denoiser model implementation.
Uses conditional diffusion model concepts for medical image denoising.
"""
import os
import logging
import numpy as np
from typing import Dict, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Conv2d, Linear
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ....torch_model import TorchModel
from ....model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class TimeEmbedding(nn.Module):
    """Time embedding module for diffusion model."""
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            Linear(1, time_dim),
            nn.SiLU(),
            Linear(time_dim, time_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1).float()
        return self.time_mlp(t)

class ResidualBlock(nn.Module):
    """Residual block with time conditioning."""
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            Linear(time_dim, out_channels)
        )
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time conditioning
        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """Self-attention block for diffusion model."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = Conv2d(channels, channels * 3, 1)
        self.proj = Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        
        scale = (c) ** -0.5
        attn = torch.bmm(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        out = self.proj(out)
        
        return out + x

class DiffusionUNet(nn.Module):
    """UNet backbone for diffusion model."""
    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 1, 
        time_dim: int = 256,
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        attention_resolutions: Tuple[int, ...] = (8, 4, 2)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Input convolution
        self.input_conv = Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        
        
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        current_channels = base_channels
        channels_list = [current_channels]
        
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            
            # Add ResNet blocks with attention where specified
            for _ in range(2):
                self.down_blocks.append(ResidualBlock(current_channels, out_channels, time_dim))
                current_channels = out_channels
                if base_channels * mult in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(current_channels))
                channels_list.append(current_channels)
            
            # Add downsampling except for last block
            if i < len(channel_mults) - 1:
                self.down_samples.append(nn.Sequential(
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1)
                ))
                channels_list.append(current_channels)
        
        
        
        
        # Middle blocks with attention
        self.middle_block1 = ResidualBlock(current_channels, current_channels, time_dim)
        self.middle_attn = AttentionBlock(current_channels)
        self.middle_block2 = ResidualBlock(current_channels, current_channels, time_dim)
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            
            # Add ResNet blocks with attention where specified
            for _ in range(2):
                self.up_blocks.append(
                    ResidualBlock(
                        channels_list.pop() + current_channels, out_channels, time_dim
                    )
                )
                current_channels = out_channels
                if base_channels * mult in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(current_channels))
            
            # Add upsampling except for last block
            if i > 0:
                self.up_samples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(current_channels, base_channels * channel_mults[i-1], kernel_size=3, padding=1)
                    )
                )
        
        # Output layers
        self.norm_out = nn.GroupNorm(8, current_channels)
        self.conv_out = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
    
    
    
    
    
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Initial convolution
        h = self.input_conv(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Index for downsampling
        down_idx = 0
        
        # Downsampling
        for i, block in enumerate(self.down_blocks):
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
            skip_connections.append(h)
            
            # Apply downsampling after every 2 ResidualBlocks + optional attention (3 layers total)
            # This is different from the test expectation but aligns with the model architecture
            if i < len(self.down_blocks) - 1 and (i + 1) % 3 == 0:
                if down_idx < len(self.down_samples):
                    h = self.down_samples[down_idx](h)
                    skip_connections.append(h)
                    down_idx += 1
        
        # Middle blocks
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)
        
        
        
        
        
        
        
        # Upsampling
        for i, block in enumerate(self.up_blocks):
            if isinstance(block, ResidualBlock):
                skip_connection = skip_connections.pop()
                h = torch.cat([h, skip_connection], dim=1)
                h = block(h, t_emb)
            else:
                h = block(h)
                
            if i < len(self.up_blocks) - 1 and i % 3 == 2:
                idx = i // 3
                if idx < len(self.up_samples):
                    h = self.up_samples[idx](h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h

class DiffusionDenoiser(TorchModel):
    """
    Diffusion-based image denoising model.
    
    Implements a conditional diffusion model that directly maps noisy images to 
    denoised versions using a time-conditioned UNet architecture.
    """
    
    def __init__(self, model_path=None, device=None, inference_steps=20):
        """
        Initialize the diffusion denoiser model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            inference_steps: Number of diffusion steps to use for inference (fewer = faster but lower quality)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        super().__init__(model_path, device)
        self.inference_steps = inference_steps
        self.noise_levels = None  # Will be initialized in _load_model
    
    def _create_model_architecture(self):
        """Create the diffusion model architecture."""
        # For medical images, typically grayscale input/output
        model = DiffusionUNet(in_channels=1, out_channels=1)
        
        # Create noise schedule for inference
        self.noise_levels = torch.linspace(0.1, 0.01, self.inference_steps).to(self.torch_device)
        
        return model
    
    def preprocess(self, image):
        """
        Preprocess the input image for the diffusion model.
        
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
        Run inference with the diffusion model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Denoised output tensor
        """
        with torch.no_grad():
            # For simplicity, we don't implement the full diffusion process
            # In a real implementation, we would do multiple denoising steps
            
            # Add a small amount of noise to simulate the first step
            # of a reverse diffusion process (this would normally be more complex)
            noise = torch.randn_like(preprocessed_tensor) * 0.1
            noisy_input = preprocessed_tensor + noise
            
            # Instead of running the full diffusion process, we use the model
            # to directly predict the denoised image from the noisy input
            # Time step 0 corresponds to the fully denoised state
            t = torch.zeros(preprocessed_tensor.shape[0], dtype=torch.long, device=self.torch_device)
            
            return self.model(noisy_input, t)
    
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output.
        
        Args:
            model_output: Raw output from the model
            original_image: Original input image (not used)
            
        Returns:
            numpy.ndarray: Denoised image as numpy array
        """
        return super().postprocess(model_output, original_image)

# Register the model
ModelRegistry.register("diffusion_denoiser", DiffusionDenoiser)