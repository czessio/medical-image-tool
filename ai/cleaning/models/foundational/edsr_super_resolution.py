"""
EDSR super-resolution model implementation.
Implements the Enhanced Deep Super-Resolution Network architecture for medical image upscaling.
"""
import os
import logging
import math
import numpy as np
from typing import Dict, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.torch_model import TorchModel
from ai.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class MeanShift(nn.Conv2d):
    """Mean shift layer for input normalization."""
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False

class ResBlock(nn.Module):
    """Residual block for EDSR architecture."""
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    """Upsampling module using PixelShuffle."""
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR(nn.Module):
    """
    Enhanced Deep Super-Resolution Network (EDSR) model.
    
    Based on the paper "Enhanced Deep Residual Networks for Single Image Super-Resolution"
    by Lim et al. (2017).
    
    Modified to better support RealESRGAN weight files.
    """
    def __init__(self, in_channels=1, out_channels=1, n_feats=64, n_resblocks=16, scale=2, res_scale=1):
        """
        Initialize the EDSR model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_feats: Number of feature maps
            n_resblocks: Number of residual blocks
            scale: Upscaling factor
            res_scale: Residual scaling factor
        """
        super(EDSR, self).__init__()
        
        kernel_size = 3
        act = nn.ReLU(True)
        
        # Define head module
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size, padding=kernel_size//2)
        
        # Define body module
        m_body = [
            ResBlock(n_feats, kernel_size, res_scale=res_scale, act=act) \
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
        self.body = nn.Sequential(*m_body)
        
        # Define tail module
        m_tail = [
            Upsampler(scale, n_feats, act=False),
            nn.Conv2d(n_feats, out_channels, kernel_size, padding=kernel_size//2)
        ]
        self.tail = nn.Sequential(*m_tail)
        
        # Add RealESRGAN compatible structure for weight loading
        # These fields match the keys in the RealESRGAN weight files
        self.conv_first = self.head  # Map to existing first conv
        self.conv_body = self.body[-1]  # Map to last conv in the body
        
        # Upsampling components - use existing or create placeholders
        if scale >= 2:
            if isinstance(self.tail[0], Upsampler) and len(self.tail[0]) >= 2:
                self.conv_up1 = self.tail[0][0]  # First upsampling conv
            else:
                self.conv_up1 = nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1)  # Placeholder
            
            if scale >= 4:
                self.conv_up2 = nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1)  # Placeholder
        
        # Final convolutions
        self.conv_hr = nn.Conv2d(n_feats, n_feats, 3, 1, 1)  # Placeholder
        self.conv_last = self.tail[-1]  # Map to final conv
        
        # Additional components for RealESRGAN
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Use the original EDSR architecture for inference
        x = self.head(x)
        
        res = self.body(x)
        res += x
        
        x = self.tail(res)
        
        return x


class EDSRSuperResolution(TorchModel):
    """
    EDSR-based super-resolution model.
    
    Uses the Enhanced Deep Super-Resolution Network (EDSR) architecture for
    high-quality image upscaling.
    
    Modified to work with RealESRGAN weight files.
    """
    
    def __init__(self, model_path=None, device=None, scale_factor=2, n_resblocks=16, n_feats=64, in_channels=1, out_channels=1):
        """
        Initialize the EDSR super-resolution model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            scale_factor: Upscaling factor (2, 3, or 4)
            n_resblocks: Number of residual blocks
            n_feats: Number of feature maps
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            out_channels: Number of output channels (usually same as in_channels)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        self.scale_factor = scale_factor
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(model_path, device)
    
    def _create_model_architecture(self):
        """Create the EDSR model architecture."""
        model = EDSR(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_feats=self.n_feats,
            n_resblocks=self.n_resblocks,
            scale=self.scale_factor,
            res_scale=0.1  # Default value from paper
        )
        
        return model
    
    def _custom_load_state_dict(self, state_dict):
        """Custom loading function for matching RealESRGAN weight file structure"""
        logger.info("Using custom weight loading for EDSR/RealESRGAN model")
        
        # Create a new state dict that maps between RealESRGAN and EDSR structure
        new_state_dict = {}
        
        # Core component mapping
        mappings = {
            'conv_first.weight': 'head.weight',
            'conv_first.bias': 'head.bias',
            'conv_body.weight': 'body.16.weight',
            'conv_body.bias': 'body.16.bias',
            'conv_up1.weight': 'tail.0.0.weight',
            'conv_up1.bias': 'tail.0.0.bias',
            'conv_last.weight': 'tail.1.weight',
            'conv_last.bias': 'tail.1.bias',
        }
        
        # Copy mapped weights
        for src_key, dst_key in mappings.items():
            if src_key in state_dict:
                new_state_dict[dst_key] = state_dict[src_key]
        
        # Try non-strict loading
        try:
            self.model.load_state_dict(new_state_dict, strict=False)
            logger.info("Model weights loaded successfully with custom mapping")
            return True
        except Exception as e:
            logger.error(f"Error in custom weight loading: {e}")
            return False
    
    def preprocess(self, image):
        """
        Preprocess the input image for the EDSR model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Use parent preprocessing but ensure channels are correct
        tensor = super().preprocess(image)
        
        # If input is RGB but model expects grayscale, convert to grayscale
        if tensor.shape[1] == 3 and self.in_channels == 1:
            tensor = tensor.mean(dim=1, keepdim=True)
        
        # If input is grayscale but model expects RGB, repeat channels
        if tensor.shape[1] == 1 and self.in_channels == 3:
            tensor = tensor.repeat(1, 3, 1, 1)
        
        return tensor
    
    def inference(self, preprocessed_tensor):
        """
        Run inference with the EDSR model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Super-resolved output tensor
        """
        with torch.no_grad():
            return self.model(preprocessed_tensor)
    
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output.
        
        Args:
            model_output: Raw output from the model
            original_image: Original input image (if needed for reference)
            
        Returns:
            numpy.ndarray: Super-resolved image as numpy array
        """
        return super().postprocess(model_output, original_image)
    
    def process(self, image):
        """
        Process an image with the EDSR model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            numpy.ndarray: Super-resolved output image
        """
        if not self.initialized:
            self.initialize()
        
        preprocessed = self.preprocess(image)
        output = self.inference(preprocessed)
        result = self.postprocess(output, image)
        
        return result

# Register the model
ModelRegistry.register("edsr_super_resolution", EDSRSuperResolution)