"""
ResNet-50 3D model implementation for medical image enhancement application.
Leverages pre-trained weights on 23 medical datasets for high-quality volumetric image processing.
"""
import os
import logging
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Type

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

class Conv3DWithBN(nn.Module):
    """3D convolution followed by batch normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Bottleneck3D(nn.Module):
    """
    Bottleneck block for ResNet50 3D.
    Uses 1x1x1, 3x3x3, 1x1x1 convolutional structure with residual connection.
    """
    expansion = 4  # Output channels = input channels * expansion
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out

class ResNet3D(nn.Module):
    """
    ResNet-50 architecture adapted for 3D medical images.
    Pre-trained on 23 medical imaging datasets.
    """
    def __init__(self, block=Bottleneck3D, layers=[3, 4, 6, 3], in_channels=1, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global pooling and final classification layer
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        # Create downsample branch if needed (when changing dimensions or stride)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
            
        layers = []
        
        # First block may have stride and downsample
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        # Update input planes for subsequent blocks
        self.inplanes = planes * block.expansion
        
        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input conv and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x1 = self.layer1(x)  # First stage output
        x2 = self.layer2(x1)  # Second stage output
        x3 = self.layer3(x2)  # Third stage output
        x4 = self.layer4(x3)  # Fourth stage output
        
        # Global pooling
        x_pool = self.avgpool(x4)
        x_flat = torch.flatten(x_pool, 1)
        
        # Classification layer (used for pre-training, may not be needed for medical tasks)
        x_out = self.fc(x_flat)
        
        # Return all intermediate activations for potential skip connections
        features = {
            'stem': x,
            'layer1': x1,
            'layer2': x2,
            'layer3': x3,
            'layer4': x4,
            'pool': x_flat,
            'out': x_out
        }
        
        return features

class ResNet50Medical(TorchModel):
    """
    ResNet-50 model for 3D medical image processing.
    Pre-trained on 23 medical datasets for various enhancement tasks.
    """
    
    def __init__(self, model_path=None, device=None, task_type='enhancement'):
        """
        Initialize the ResNet-50 medical model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            task_type: Type of task ('enhancement', 'segmentation', 'classification')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        self.task_type = task_type
        super().__init__(model_path, device)
    
    def _create_model_architecture(self):
        """Create the ResNet-50 model architecture."""
        model = ResNet3D(
            block=Bottleneck3D,
            layers=[3, 4, 6, 3],  # Standard ResNet-50 layer configuration
            in_channels=1,        # Medical images are typically grayscale
            num_classes=23        # Pre-trained on 23 medical datasets
        )
        
        return model
    
    def _custom_load_state_dict(self, state_dict):
        """
        Custom loading function for matching ResNet-50 weight structure.
        
        Args:
            state_dict: State dictionary from the weight file
            
        Returns:
            bool: True if loading succeeded, False otherwise
        """
        logger.info("Using custom weight loading for ResNet-50 medical model")
        
        try:
            # Handle the 'module.' prefix that comes from DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    # Remove the 'module.' prefix
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] = v
            
            # Load the weights with non-strict matching
            self.model.load_state_dict(new_state_dict, strict=False)
            
            # Report on missing and unexpected keys
            missing_keys = set(k for k, _ in self.model.named_parameters()) - set(new_state_dict.keys())
            unexpected_keys = set(new_state_dict.keys()) - set(k for k, _ in self.model.named_parameters())
            
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)} keys")
                logger.debug(f"Missing keys sample: {list(missing_keys)[:5]}")
            
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                logger.debug(f"Unexpected keys sample: {list(unexpected_keys)[:5]}")
            
            logger.info("Model weights loaded with non-strict matching")
            return True
            
        except Exception as e:
            logger.error(f"Error in custom weight loading: {e}")
            return False
    
    def preprocess(self, image):
        """
        Preprocess the input image for the ResNet-50 model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Use parent preprocessing but ensure the image is in the correct format
        tensor = super().preprocess(image)
        
        # If input is 2D, expand to 3D
        if len(tensor.shape) == 4:  # [B, C, H, W]
            tensor = tensor.unsqueeze(2)  # [B, C, 1, H, W]
        
        # If input is RGB, convert to grayscale
        if tensor.shape[1] == 3:
            tensor = tensor.mean(dim=1, keepdim=True)
        
        return tensor
    
    def inference(self, preprocessed_tensor):
        """
        Run inference with the ResNet-50 model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Processed feature representation
        """
        with torch.no_grad():
            # Get features from the model
            features = self.model(preprocessed_tensor)
            
            # Determine which features to use based on task type
            if self.task_type == 'enhancement':
                # For enhancement, use the penultimate layer features
                return features['layer4']
            elif self.task_type == 'segmentation':
                # For segmentation, use multiple levels of features
                return features
            else:  # classification
                # For classification, use the final output
                return features['out']
    
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output for the task.
        
        Args:
            model_output: Raw output from the model
            original_image: Original input image (if needed for reference)
            
        Returns:
            numpy.ndarray: Processed output relevant to the task
        """
        # For enhancement tasks, convert the feature maps to an enhanced image
        if self.task_type == 'enhancement':
            # Extract the features and reshape to match the original image
            output = model_output.squeeze().cpu().numpy()
            
            # For simplicity, we're returning the feature maps for now
            # In a real application, you might want to apply a decoder network
            # to transform these features into an enhanced image
            
            # If the original image is 2D but our output is 3D, take a representative slice
            if len(output.shape) > 2 and original_image is not None and len(original_image.shape) == 2:
                # Take the middle slice from the depth dimension
                if len(output.shape) == 4:  # [C, D, H, W]
                    middle_idx = output.shape[1] // 2
                    output = output[:, middle_idx, :, :]
                elif len(output.shape) == 3:  # [D, H, W]
                    middle_idx = output.shape[0] // 2
                    output = output[middle_idx]
            
            # Normalize to [0, 1] range
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            
            return output
            
        # For segmentation or classification, return appropriate outputs
        elif self.task_type == 'segmentation':
            # You would implement a decoder head for segmentation
            return None
        else:  # classification
            # Return class probabilities
            return F.softmax(model_output, dim=1).cpu().numpy()

# Register the model with the ModelRegistry
ModelRegistry.register("novel_resnet50_medical", ResNet50Medical)