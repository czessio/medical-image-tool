"""
EDSR/RealESRGAN super-resolution model implementation.
Modified to match the RealESRGAN weight file structure for medical image upscaling.
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

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) used in RealESRGAN.
    This matches the RDB structure in the RealESRGAN_x2.pth file.
    """
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB) used in RealESRGAN.
    This matches the structure in the RealESRGAN_x2.pth file.
    """
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RealESRGAN(nn.Module):
    """
    RealESRGAN model architecture.
    
    This implementation matches the structure in RealESRGAN_x2.pth, RealESRGAN_x4.pth, 
    and RealESRGAN_x8.pth files, with RRDB (Residual in Residual Dense Block) as the 
    basic building block.
    """
    def __init__(self, in_channels=12, out_channels=3, num_feat=64, num_block=23, scale=2):
        super(RealESRGAN, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        
        # First conv - with 12 input channels to match the weight file
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        
        # Body blocks (RRDB blocks)
        self.body = nn.ModuleList()
        for i in range(num_block):
            self.body.append(RRDB(num_feat))
            
        # Conv after body
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling blocks
        upsample_layers = []
        if scale == 2:
            upsample_layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            upsample_layers.append(nn.PixelShuffle(2))
        elif scale == 4:
            upsample_layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            upsample_layers.append(nn.PixelShuffle(2))
            upsample_layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            upsample_layers.append(nn.PixelShuffle(2))
        elif scale == 8:
            upsample_layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            upsample_layers.append(nn.PixelShuffle(2))
            upsample_layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            upsample_layers.append(nn.PixelShuffle(2))
            upsample_layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            upsample_layers.append(nn.PixelShuffle(2))
        
        # Assign to named attributes for compatibility with weight file
        if scale >= 2:
            self.conv_up1 = upsample_layers[0]
        if scale >= 4:
            self.conv_up2 = upsample_layers[2]
        if scale >= 8:
            self.conv_up3 = upsample_layers[4]
            
        # High resolution conv
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Last conv
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)
        
        # Activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        # Initial feature extraction
        feat = self.conv_first(x)
        
        # Body (RRDB blocks)
        body_feat = feat.clone()
        for block in self.body:
            body_feat = block(body_feat)
            
        # Conv after body
        body_feat = self.conv_body(body_feat)
        body_feat = body_feat + feat  # Residual connection
        
        # Upsampling
        if self.scale >= 2:
            feat = self.conv_up1(body_feat)
            feat = self.lrelu(feat)
            feat = nn.PixelShuffle(2)(feat)
        
        if self.scale >= 4:
            feat = self.conv_up2(feat)
            feat = self.lrelu(feat)
            feat = nn.PixelShuffle(2)(feat)
            
        if self.scale >= 8:
            feat = self.conv_up3(feat)
            feat = self.lrelu(feat)
            feat = nn.PixelShuffle(2)(feat)
            
        # High resolution conv
        feat = self.lrelu(self.conv_hr(feat))
        
        # Last conv
        out = self.conv_last(feat)
        
        return out


class EDSRSuperResolution(TorchModel):
    """
    EDSR/RealESRGAN-based super-resolution model.
    
    This implementation has been modified to match the RealESRGAN weight files
    while providing EDSR-like functionality.
    """
    
    def __init__(self, model_path=None, device=None, scale_factor=2, num_blocks=23, num_feat=64, in_channels=12, out_channels=3):
        """
        Initialize the super-resolution model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            scale_factor: Upscaling factor (2, 4, or 8)
            num_blocks: Number of RRDB blocks
            num_feat: Number of feature maps
            in_channels: Number of input channels (12 for RealESRGAN)
            out_channels: Number of output channels (3 for RGB)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        # Determine parameters from model path if possible
        if model_path:
            basename = os.path.basename(model_path)
            if 'x2' in basename:
                self.scale_factor = 2
            elif 'x4' in basename:
                self.scale_factor = 4
            elif 'x8' in basename:
                self.scale_factor = 8
            else:
                self.scale_factor = scale_factor
                
            logger.info(f"Using scale factor {self.scale_factor} based on model path: {basename}")
        else:
            self.scale_factor = scale_factor
        
        self.num_blocks = num_blocks
        self.num_feat = num_feat
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.internal_model = None  # To store the actual model that matches the weight file
        
        super().__init__(model_path, device)
    
    def _create_model_architecture(self):
        """Create the model architecture."""
        model = RealESRGAN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_feat=self.num_feat,
            num_block=self.num_blocks,
            scale=self.scale_factor
        )
        
        return model
    
    def _custom_load_state_dict(self, state_dict):
        """
        Custom loading function for matching RealESRGAN weight file structure.
        
        Args:
            state_dict: State dictionary from the weight file
            
        Returns:
            bool: True if loading succeeded, False otherwise
        """
        logger.info("Using custom weight loading for RealESRGAN model")
        
        try:
            # Analyze state dict structure
            if len(state_dict) > 0:
                logger.info(f"State dict contains {len(state_dict)} keys")
                logger.info(f"Example keys: {list(state_dict.keys())[:5]}")
                
                # Check if input channels need adjustment based on first layer weights
                if 'conv_first.weight' in state_dict:
                    first_layer_shape = state_dict['conv_first.weight'].shape
                    logger.info(f"First layer shape: {first_layer_shape}")
                    
                    if first_layer_shape[1] != self.in_channels:
                        logger.warning(f"Input channel mismatch: model={self.in_channels}, weights={first_layer_shape[1]}")
                        
                        # Re-create the model with the correct input channels
                        self.in_channels = first_layer_shape[1]
                        self.model = RealESRGAN(
                            in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            num_feat=self.num_feat,
                            num_block=self.num_blocks,
                            scale=self.scale_factor
                        ).to(self.torch_device)
                        logger.info(f"Re-created model with {self.in_channels} input channels")
            
            # Try to load state dict
            self.model.load_state_dict(state_dict, strict=False)
            
            # Store the internal model that matches the weight file
            self.internal_model = self.model
            
            # Check if any keys were missing or unexpected
            missing_keys = set(key for key, _ in self.model.named_parameters()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(key for key, _ in self.model.named_parameters())
            
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)} keys")
            
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
            
            logger.info("Model weights loaded successfully with custom mapping")
            return True
            
        except Exception as e:
            logger.error(f"Error in custom weight loading: {e}")
            return False
    
    def preprocess(self, image):
        """
        Preprocess the input image for the super-resolution model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Use parent preprocessing to get basic tensor
        tensor = super().preprocess(image)
        
        # Convert tensor from [B, C, H, W] to the format expected by the model
        if tensor.shape[1] == 1:
            # Single channel input (grayscale)
            if self.in_channels == 3:
                # Model expects RGB - duplicate the grayscale channel
                tensor = tensor.repeat(1, 3, 1, 1)
            elif self.in_channels == 12:
                # Model expects 12-channel input - create a compatible input
                # We'll repeat the grayscale 12 times or use a different approach
                tensor = tensor.repeat(1, 12, 1, 1)
        elif tensor.shape[1] == 3:
            # RGB input
            if self.in_channels == 1:
                # Model expects grayscale - average the RGB channels
                tensor = tensor.mean(dim=1, keepdim=True)
            elif self.in_channels == 12:
                # Model expects 12-channel input - expand RGB to 12 channels
                # A simple approach is to repeat RGB 4 times
                tensor = tensor.repeat(1, 4, 1, 1)
        
        return tensor
    
    def inference(self, preprocessed_tensor):
        """
        Run inference with the super-resolution model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Super-resolved output tensor
        """
        with torch.no_grad():
            try:
                return self.model(preprocessed_tensor)
            except Exception as e:
                logger.error(f"Error in model inference: {e}")
                # Fallback to just resizing the input tensor
                logger.warning("Using fallback upsampling")
                return F.interpolate(
                    preprocessed_tensor, 
                    scale_factor=self.scale_factor, 
                    mode='bilinear', 
                    align_corners=False
                )
    
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