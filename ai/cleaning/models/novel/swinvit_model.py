"""
SwinViT-based model implementation for the medical image enhancement application.
Leverages the Swin Transformer architecture for high-quality volumetric image processing.
"""
import os
import logging
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Union

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

# Helper functions for Swin Transformer
def window_partition(x, window_size):
    """
    Partition into non-overlapping windows
    Args:
        x: (B, D, H, W, C)
        window_size: tuple of window size for D, H, W
    Returns:
        windows: (B*num_windows, window_size_d, window_size_h, window_size_w, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, 
               D // window_size[0], window_size[0], 
               H // window_size[1], window_size[1], 
               W // window_size[2], window_size[2], 
               C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """
    Reverse window partition
    Args:
        windows: (B*num_windows, window_size_d, window_size_h, window_size_w, C)
        window_size: tuple of window size for D, H, W
        B, D, H, W: original tensor dimensions
    Returns:
        x: (B, D, H, W, C)
    """
    C = windows.shape[-1]
    x = windows.view(B, 
                    D // window_size[0], H // window_size[1], W // window_size[2], 
                    window_size[0], window_size[1], window_size[2], C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, C)
    return x

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding layer for Swin Transformer.
    """
    def __init__(self, patch_size=(2, 2, 2), in_channels=1, embed_dim=48):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Projection layer that maps input to embedding dimension
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                             kernel_size=patch_size, 
                             stride=patch_size)
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Input: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Ensure input dimensions are divisible by patch size
        pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        
        # Apply projection
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        
        # Reshape and transpose for LayerNorm
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # (B, D'*H'*W', embed_dim)
        x = self.norm(x)
        
        return x, (D, H, W)

class WindowAttention3D(nn.Module):
    """
    Window-based Multi-head Self-Attention module for 3D data.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Define relative position bias table
        # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * 
                        (2 * window_size[1] - 1) * 
                        (2 * window_size[2] - 1), num_heads))
        
        # Get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        
        # Calculate relative positions
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape (num_windows*B, N, C)
            mask: (0/-inf) mask with shape (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, C//num_heads
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        # Apply relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x

class SwinTransformerBlock3D(nn.Module):
    """
    3D Swin Transformer Block
    """
    def __init__(self, dim, num_heads, window_size=(7, 7, 7),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias)
        
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x, mask=None):
        # Extract dimensions from input tensor
        B, D, H, W, C = x.shape  # Add this line
        
        shortcut = x
        x = self.norm1(x)
        
        # Window attention
        x_windows = window_partition(x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], self.dim)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask)  
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], self.dim)
        x = window_reverse(attn_windows, self.window_size, B, D, H, W)  # Now B, D, H, W are defined
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class Mlp(nn.Module):
    """MLP as used in Swin Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinViT(nn.Module):
    """
    Swin Transformer for 3D medical image processing
    """
    def __init__(self, 
                 in_channels=1,
                 embed_dim=48,
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_size=(2, 2, 2),
                 use_checkpoint=False,
                 output_dim=1):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Swin Transformer layers
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=embed_dim * 2**i_layer,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                ) for _ in range(depths[i_layer])
            ])
            
            # Downsample layer at the end of each stage
            if i_layer < self.num_layers - 1:
                layer.append(nn.Sequential(
                    norm_layer(embed_dim * 2**i_layer),
                    nn.Linear(embed_dim * 2**i_layer, embed_dim * 2**(i_layer+1)),
                ))
            
            self.layers.append(layer)
        
        # Final normalization
        self.norm = norm_layer(embed_dim * 2**(self.num_layers-1))
        
        # Task-specific heads
        self.rotation_head = nn.Linear(embed_dim * 2**(self.num_layers-1), 4)  # Quaternion representation
        self.contrastive_head = nn.Linear(embed_dim * 2**(self.num_layers-1), 512)
        
        # 3D transposed convolution for output
        self.convTrans3d = nn.ConvTranspose3d(embed_dim * 2**(self.num_layers-1), output_dim, 
                                             kernel_size=32, stride=32)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Patch embedding
        x, (D, H, W) = self.patch_embed(x)  # (B, L, C)
        B = x.shape[0]
        
        # View as 3D volume
        x = x.view(B, D, H, W, self.embed_dim)
        
        # Process through Swin Transformer blocks
        features = []
        for i, layer_blocks in enumerate(self.layers):
            # Process all blocks in the current layer
            for j, block in enumerate(layer_blocks):
                if j == len(layer_blocks) - 1 and i < self.num_layers - 1:
                    # This is a downsample layer
                    x = x.view(B, -1, x.shape[-1])
                    x = block(x)
                    D, H, W = D//2, H//2, W//2  # Update spatial dimensions
                    x = x.view(B, D, H, W, x.shape[-1])
                else:
                    # This is a transformer block
                    x = block(x)
            
            # Store feature at the end of each layer
            features.append(x)
        
        # Final normalization
        x = self.norm(x.view(B, -1, x.shape[-1]))
        
        # Generate task-specific outputs
        rotation_output = self.rotation_head(x.mean(dim=1))
        contrastive_output = self.contrastive_head(x.mean(dim=1))
        
        # Reshape for transposed convolution
        x = x.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        volume_output = self.convTrans3d(x)
        
        return {
            'volume': volume_output,
            'rotation': rotation_output,
            'contrastive': contrastive_output,
            'features': features
        }

class SwinViTModel(TorchModel):
    """
    SwinViT model for 3D medical image processing.
    """
    
    def __init__(self, model_path=None, device=None, task_type='reconstruction'):
        """
        Initialize the SwinViT model.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            task_type: Type of task ('reconstruction', 'segmentation', or 'enhancement')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        self.task_type = task_type
        super().__init__(model_path, device)
    
    def _create_model_architecture(self):
        """Create the SwinViT model architecture."""
        model = SwinViT(
            in_channels=1,      # Medical images are typically grayscale
            embed_dim=48,       # Match the weight structure
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(7, 7, 7),
            output_dim=1        # Output the same number of channels as input
        )
        
        return model
    
    def _custom_load_state_dict(self, state_dict):
        """
        Custom loading function for matching SwinViT weight structure.
        
        Args:
            state_dict: State dictionary from the weight file
            
        Returns:
            bool: True if loading succeeded, False otherwise
        """
        logger.info("Using custom weight loading for SwinViT model")
        
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
        Preprocess the input image for the SwinViT model.
        
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
        Run inference with the SwinViT model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Output tensor with processed image
        """
        with torch.no_grad():
            outputs = self.model(preprocessed_tensor)
            
            # Extract the appropriate output based on task type
            if self.task_type == 'reconstruction':
                return outputs['volume']
            elif self.task_type == 'segmentation':
                return torch.sigmoid(outputs['volume'])
            else:  # enhancement
                return outputs['volume']
    
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output.
        
        Args:
            model_output: Raw output from the model
            original_image: Original input image (if needed for reference)
            
        Returns:
            numpy.ndarray: Processed output image
        """
        # Move tensor to CPU and convert to numpy
        output = model_output.squeeze().cpu().numpy()
        
        # Ensure output values are in [0, 1]
        output = np.clip(output, 0, 1)
        
        # If the original image is 2D but our output is 3D, take the middle slice
        if len(output.shape) == 3 and original_image is not None and len(original_image.shape) == 2:
            middle_idx = output.shape[0] // 2
            output = output[middle_idx]
        
        return output

# Register the model with the ModelRegistry
ModelRegistry.register("novel_swinvit", SwinViTModel)