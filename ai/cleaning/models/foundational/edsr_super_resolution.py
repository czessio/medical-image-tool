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
        
    def forward(self, x):
        """Forward pass through the network."""
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
    """
    
    def __init__(self, model_path=None, device=None, scale_factor=2, n_resblocks=16, n_feats=64):
        """
        Initialize the EDSR super-resolution model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            scale_factor: Upscaling factor (2, 3, or 4)
            n_resblocks: Number of residual blocks
            n_feats: Number of feature maps
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        self.scale_factor = scale_factor
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        super().__init__(model_path, device)
    
    def _create_model_architecture(self):
        """Create the EDSR model architecture."""
        # For medical images, typically grayscale input/output
        model = EDSR(
            in_channels=1,
            out_channels=1,
            n_feats=self.n_feats,
            n_resblocks=self.n_resblocks,
            scale=self.scale_factor,
            res_scale=0.1  # Default value from paper
        )
        
        return model
    
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
        
        # If input is RGB, convert to grayscale (average channels)
        if tensor.shape[1] == 3:
            tensor = tensor.mean(dim=1, keepdim=True)
        
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

# Register the model
ModelRegistry.register("edsr_super_resolution", EDSRSuperResolution)