"""
U-Net artifact removal model implementation.
Implements the classic U-Net architecture for medical image artifact removal.
"""
import os
import logging
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

class DoubleConv(nn.Module):
    """(convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Full U-Net architecture for image artifact removal.
    
    Based on the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by Ronneberger et al. (2015).
    """
    def __init__(self, in_channels=1, out_channels=1, n_channels=64, bilinear=True):
        """
        Initialize the U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_channels: Number of channels in the first layer
            bilinear: Whether to use bilinear upsampling or transposed convolutions
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, n_channels)
        self.down1 = Down(n_channels, n_channels * 2)
        self.down2 = Down(n_channels * 2, n_channels * 4)
        self.down3 = Down(n_channels * 4, n_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_channels * 8, n_channels * 16 // factor)
        self.up1 = Up(n_channels * 16, n_channels * 8 // factor, bilinear)
        self.up2 = Up(n_channels * 8, n_channels * 4 // factor, bilinear)
        self.up3 = Up(n_channels * 4, n_channels * 2 // factor, bilinear)
        self.up4 = Up(n_channels * 2, n_channels, bilinear)
        self.outc = OutConv(n_channels, out_channels)

    def forward(self, x):
        """Forward pass through the U-Net."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Apply sigmoid to output for artifact removal (maps to 0-1 range)
        return torch.sigmoid(logits)


class UNetArtifactRemoval(TorchModel):
    """
    U-Net-based artifact removal model.
    
    Uses the classic U-Net architecture which was originally designed for
    biomedical image segmentation but works well for artifact removal due to
    its encoder-decoder structure with skip connections.
    """
    
    def __init__(self, model_path=None, device=None, n_channels=64, bilinear=True):
        """
        Initialize the U-Net artifact removal model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            n_channels: Number of channels in the first layer
            bilinear: Whether to use bilinear upsampling or transposed convolutions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        self.n_channels = n_channels
        self.bilinear = bilinear
        super().__init__(model_path, device)
    
    def _create_model_architecture(self):
        """Create the U-Net model architecture."""
        # For medical images, typically grayscale input/output
        model = UNet(
            in_channels=1,
            out_channels=1,
            n_channels=self.n_channels,
            bilinear=self.bilinear
        )
        
        return model
    
    def preprocess(self, image):
        """
        Preprocess the input image for the U-Net model.
        
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
        Run inference with the U-Net model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Artifact-free output tensor
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
            numpy.ndarray: Artifact-free image as numpy array
        """
        return super().postprocess(model_output, original_image)

# Register the model
ModelRegistry.register("unet_artifact_removal", UNetArtifactRemoval)