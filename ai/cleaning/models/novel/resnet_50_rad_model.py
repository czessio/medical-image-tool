"""
ResNet-50 model implementation pretrained on RadImageNet for medical image enhancement.
Implements a standard 2D ResNet architecture optimized for radiological image analysis.
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

class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet50.
    Uses 1x1, 3x3, 1x1 convolutional structure with residual connection.
    """
    expansion = 4  # Output channels = input channels * expansion
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    """
    ResNet architecture for 2D medical image processing.
    Structured to match RadImageNet's pretrained weights.
    """
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], in_channels=3, num_classes=1000):
        super().__init__()
        
        # Create a sequential model to match backbone.X.Y structure
        self.backbone = nn.ModuleList()
        
        # Initial convolution
        self.backbone.append(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.backbone.append(nn.BatchNorm2d(64))
        self.backbone.append(nn.ReLU(inplace=True))
        self.backbone.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # ResNet Layers (4.X = layer1, 5.X = layer2, etc.)
        self.inplanes = 64
        self.backbone.append(self._make_layer(block, 64, layers[0]))
        self.backbone.append(self._make_layer(block, 128, layers[1], stride=2))
        self.backbone.append(self._make_layer(block, 256, layers[2], stride=2))
        self.backbone.append(self._make_layer(block, 512, layers[3], stride=2))
        
        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        # Create downsample branch if needed
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
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
        # Initial convolution and pooling
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        
        # ResNet blocks - collecting features at each stage
        features = {}
        
        x = self.backbone[4](x)  # Layer 1
        features['layer1'] = x
        
        x = self.backbone[5](x)  # Layer 2
        features['layer2'] = x
        
        x = self.backbone[6](x)  # Layer 3
        features['layer3'] = x
        
        x = self.backbone[7](x)  # Layer 4
        features['layer4'] = x
        
        # Global pooling
        x = self.avgpool(x)
        x_flat = torch.flatten(x, 1)
        
        # Classification
        x_out = self.fc(x_flat)
        
        # Add additional outputs to the feature dictionary
        features['pool'] = x_flat
        features['out'] = x_out
        
        return features

class ResNet50RadImageNet(TorchModel):
    """
    ResNet-50 model pretrained on RadImageNet for 2D medical image processing.
    Provides features specialized for radiological image analysis.
    """
    
    def __init__(self, model_path=None, device=None, task_type='enhancement'):
        """
        Initialize the ResNet-50 RadImageNet model.
        
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
        """Create the ResNet-50 model architecture for RadImageNet weights."""
        model = ResNet(
            block=Bottleneck,
            layers=[3, 4, 6, 3],  # Standard ResNet-50 layer configuration
            in_channels=3,        # RadImageNet uses RGB images (3 channels)
            num_classes=1000      # Standard ImageNet/RadImageNet classes
        )
        
        return model
    
    def _custom_load_state_dict(self, state_dict):
        """
        Custom loading function for matching RadImageNet ResNet-50 weight structure.
        
        Args:
            state_dict: State dictionary from the weight file
            
        Returns:
            bool: True if loading succeeded, False otherwise
        """
        logger.info("Using custom weight loading for ResNet-50 RadImageNet model")
        
        try:
            # Try to load with non-strict matching
            self.model.load_state_dict(state_dict, strict=False)
            
            # Report on missing and unexpected keys
            missing_keys = set(k for k, _ in self.model.named_parameters()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(k for k, _ in self.model.named_parameters())
            
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)} keys")
                logger.debug(f"Missing keys sample: {list(missing_keys)[:5]}")
            
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                logger.debug(f"Unexpected keys sample: {list(unexpected_keys)[:5]}")
            
            logger.info("Model weights loaded with non-strict matching")
            return True
            
        except Exception as e:
            logger.error(f"Error in weight loading: {e}")
            return False
    
    def preprocess(self, image):
        """
        Preprocess the input image for the ResNet-50 RadImageNet model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Use parent preprocessing but ensure the image is in the correct format
        tensor = super().preprocess(image)
        
        # If input is grayscale, convert to RGB by repeating channel
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
        tensor = (tensor - mean) / std
        
        return tensor
    
    def inference(self, preprocessed_tensor):
        """
        Run inference with the ResNet-50 RadImageNet model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor or Dict: Processed feature representation or outputs
        """
        with torch.no_grad():
            # Get features from the model
            features = self.model(preprocessed_tensor)
            
            # Determine which features to use based on task type
            if self.task_type == 'enhancement':
                # For enhancement, we can use the high-level features from layer 4
                return features['layer4']
            elif self.task_type == 'segmentation':
                # For segmentation, we might want features from multiple levels
                return features
            else:  # classification
                # For classification, use the final output
                return features['out']
    
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output for the specific task.
        
        Args:
            model_output: Raw output from the model
            original_image: Original input image (if needed for reference)
            
        Returns:
            numpy.ndarray: Processed output relevant to the task
        """
        if self.task_type == 'enhancement':
            # For enhancement, convert features to a display-friendly output
            # This would typically involve a decoder network, but we'll simplify for now
            
            # If model_output is a dictionary, get the layer4 features
            if isinstance(model_output, dict):
                features = model_output['layer4']
            else:
                features = model_output
            
            # Convert to numpy and take first feature channels for visualization
            output = features.squeeze().cpu().numpy()
            
            # For visualization, take average across feature channels
            if len(output.shape) > 2:
                output = np.mean(output, axis=0)
            
            # Normalize to [0, 1] range for display
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            
            # Resize to match original image if needed
            if original_image is not None and output.shape != original_image.shape[:2]:
                from data.processing.transforms import resize_image
                output = resize_image(output, (original_image.shape[1], original_image.shape[0]))
            
            return output
            
        elif self.task_type == 'classification':
            # Return class probabilities
            return F.softmax(model_output, dim=1).cpu().numpy()
        
        else:  # segmentation or other tasks
            # For segmentation, you'd need to implement a decoder head
            logger.warning("Segmentation task not fully implemented for RadImageNet ResNet-50")
            return None

# Register the model with the ModelRegistry
ModelRegistry.register("novel_resnet50_rad", ResNet50RadImageNet)