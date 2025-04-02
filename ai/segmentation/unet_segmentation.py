# File: ai/segmentation/unet_segmentation.py

"""
U-Net segmentation model implementation for medical image enhancement application.
"""
import os
import logging
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.segmentation.segmentation_model import SegmentationModel
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

class UNetSegmentation(nn.Module):
    """
    U-Net segmentation model architecture.
    
    Based on the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by Ronneberger et al. (2015).
    """
    def __init__(self, in_channels=1, num_classes=2, n_channels=64, bilinear=True):
        """
        Initialize the U-Net model.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            n_channels: Number of channels in the first layer
            bilinear: Whether to use bilinear upsampling or transposed convolutions
        """
        super(UNetSegmentation, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
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
        self.outc = OutConv(n_channels, num_classes)

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
        
        return logits

class UNetSegmentationModel(SegmentationModel):
    """
    U-Net-based segmentation model.
    
    Uses the classic U-Net architecture which is well-suited for
    medical image segmentation tasks.
    """
    
    def __init__(self, model_path=None, device=None, num_classes=2, n_channels=64):
        """
        Initialize the U-Net segmentation model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            num_classes: Number of segmentation classes
            n_channels: Number of channels in the first layer
        """
        self.n_channels = n_channels
        super().__init__(model_path, device, num_classes)
    
    def _create_model_architecture(self):
        """Create the U-Net model architecture."""
        model = UNetSegmentation(
            in_channels=1,  # Grayscale input
            num_classes=self.num_classes,
            n_channels=self.n_channels,
            bilinear=True
        )
        
        return model
    
    def inference(self, preprocessed_tensor):
        """
        Run inference with the U-Net model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Segmentation output tensor
        """
        with torch.no_grad():
            output = self.model(preprocessed_tensor)
            
            # Apply softmax for multi-class segmentation
            if self.num_classes > 1:
                output = F.softmax(output, dim=1)
            else:
                output = torch.sigmoid(output)
                
            return output

# Register the model
ModelRegistry.register("unet_segmentation", UNetSegmentationModel)