"""
Vision Transformer (ViT) model with Masked Autoencoder (MAE) pretraining on chest X-rays.
Provides powerful feature extraction and image reconstruction capabilities for medical images.
"""
import os
import logging
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Type
from functools import partial


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import LayerNorm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.torch_model import TorchModel
from ai.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for Vision Transformer.
    Converts image to sequence of patch embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # Reshape: [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        x = self.proj(x)
        
        # Flatten: [B, embed_dim, H/patch_size, W/patch_size] -> [B, embed_dim, H*W/patch_size^2]
        x = x.flatten(2)
        
        # Transpose: [B, embed_dim, H*W/patch_size^2] -> [B, H*W/patch_size^2, embed_dim]
        x = x.transpose(1, 2)
        
        return x

class Attention(nn.Module):
    """
    Multi-head self-attention module for Vision Transformer.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    """
    MLP module for Vision Transformer.
    Two-layer MLP with GELU activation.
    """
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

class Block(nn.Module):
    """
    Transformer encoder block for Vision Transformer.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformerMAE(nn.Module):
    """
    Vision Transformer with Masked Autoencoder (MAE) pretraining.
    Includes both encoder and decoder components.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        decoder_embed_dim=512,
        depth=12,
        decoder_depth=8,
        num_heads=12,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False
    ):
        super().__init__()
        
        # ---------- Encoder ----------
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        # Normalization layer
        self.norm = norm_layer(embed_dim)
        
        # ---------- Decoder ----------
        # Decoder embedding
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder position embeddings
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim))
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        
        # Decoder normalization
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Decoder prediction layer
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_channels, bias=True)
        
        # Initialize weights
        self.initialize_weights()
        
        # Loss normalization for reconstruction
        self.norm_pix_loss = norm_pix_loss
    
    def initialize_weights(self):
        # Initialize patch_embed, cls_token, mask_token, pos_embed like in MAE paper
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize position embeddings with sine-cosine embeddings
        pos_embed = self.get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = self.get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
        """
        Generate 2D sine-cosine positional embedding.
        grid_size: int of the grid height and width
        embed_dim: output dimension for each position
        return: [grid_size*grid_size, embed_dim] if cls_token is False, 
               [1+grid_size*grid_size, embed_dim] if cls_token is True
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # Here we reverse the order since we want a transpose
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        
        # Create positional encoding
        pos_embed = np.zeros([1, grid_size, grid_size, embed_dim], dtype=np.float32)
        pos_embed[0, :, :, :embed_dim//2] = np.repeat(
            np.expand_dims(
                VisionTransformerMAE.get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[0])
            , axis=1),
            grid_size, axis=1
        )
        pos_embed[0, :, :, embed_dim//2:] = np.repeat(
            np.expand_dims(
                VisionTransformerMAE.get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[1])
            , axis=0), 
            grid_size, axis=0
        )
        pos_embed = pos_embed.reshape([1, grid_size*grid_size, embed_dim])
        
        # Add classification token position embedding
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, 1, embed_dim]), pos_embed], axis=1)
        
        return pos_embed[0]
    
    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """
        Get 1D sin-cos positional embedding from grid.
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        return: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)
        
        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2)
        
        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)
        
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb
    
    def forward_encoder(self, x, mask_ratio=0):
        """
        Forward pass of the encoder.
        x: input image, shape [B, C, H, W]
        mask_ratio: mask ratio for MAE. 0 means no masking.
        """
        # Embed patches [B, C, H, W] -> [B, N, D]
        x = self.patch_embed(x)
        
        # Add positional encoding to patches
        # [B, N, D] -> [B, N+1, D]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        return x
    
    def forward_decoder(self, x, mask_ratio=0):
        """
        Forward pass of the decoder for reconstruction.
        x: encoded tokens from the encoder
        mask_ratio: mask ratio (for actual inference, this should be 0)
        """
        # Embedding tokens from encoder to decoder 
        x = self.decoder_embed(x)
        
        # Add decoder positional encoding
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        # Apply final normalization
        x = self.decoder_norm(x)
        
        # Predict original pixels (patch-wise)
        x = self.decoder_pred(x)
        
        # Return only patch predictions (remove cls token prediction)
        return x[:, 1:, :]
    
    def forward(self, x, mask_ratio=0, encoder_only=False):
        """
        Forward pass of the full model.
        x: input image, shape [B, C, H, W]
        mask_ratio: mask ratio for MAE. 0 means no masking.
        encoder_only: if True, only encoder features are returned
        """
        # Get encoded representations
        latent = self.forward_encoder(x, mask_ratio)
        
        if encoder_only:
            return latent
        
        # Decode features for reconstruction
        pred = self.forward_decoder(latent, mask_ratio)
        
        # Reshape prediction to match original image patches
        B, N, C = pred.shape
        p = int(self.patch_embed.patch_size)
        h = w = int(self.patch_embed.grid_size)
        
        pred = pred.reshape(B, h, w, p, p, -1)
        pred = pred.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
        pred = pred.reshape(B, -1, h * p, w * p)  # [B, C, H, W]
        
        return {"latent": latent, "reconstruction": pred}

class ViTMAECXR(TorchModel):
    """
    Vision Transformer with MAE pretraining on chest X-rays.
    Can be used for both feature extraction and image reconstruction.
    """
    
    def __init__(self, model_path=None, device=None, task_type='enhancement', encoder_only=False):
        """
        Initialize the ViT-MAE model for chest X-rays.
        
        Args:
            model_path: Path to model weights file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            task_type: Type of task ('enhancement', 'classification', 'segmentation', 'reconstruction')
            encoder_only: If True, only use the encoder part of the model for feature extraction
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        self.task_type = task_type
        self.encoder_only = encoder_only
        super().__init__(model_path, device)
    
    def _create_model_architecture(self):
        """Create the ViT-MAE model architecture."""
        model = VisionTransformerMAE(
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            decoder_embed_dim=512,
            depth=12,
            decoder_depth=8,
            num_heads=12,
            decoder_num_heads=8,
            mlp_ratio=4.0
        )
        
        return model
    
    def _custom_load_state_dict(self, state_dict):
        """
        Custom loading function for matching ViT-MAE weight structure.
        
        Args:
            state_dict: State dictionary from the weight file
            
        Returns:
            bool: True if loading succeeded, False otherwise
        """
        logger.info("Using custom weight loading for ViT-MAE CXR model")
        
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
        Preprocess the input image for the ViT-MAE model.
        
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
        
        # Resize to 224x224 if needed
        if tensor.shape[2:] != (224, 224):
            tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize for pretrained model
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
        tensor = (tensor - mean) / std
        
        return tensor
    
    def inference(self, preprocessed_tensor):
        """
        Run inference with the ViT-MAE model.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
            
        Returns:
            dict: Model outputs including latent features and possibly reconstruction
        """
        with torch.no_grad():
            # Run inference either with encoder only or full model
            if self.encoder_only or self.task_type in ['classification', 'segmentation']:
                latent = self.model.forward_encoder(preprocessed_tensor)
                return {"latent": latent}
            else:
                # Full model inference (encoder + decoder)
                result = self.model(preprocessed_tensor, mask_ratio=0)
                return result
    
    def postprocess(self, model_output, original_image=None):
        """
        Postprocess the model output for the specific task.
        
        Args:
            model_output: Raw output from the model
            original_image: Original input image (if needed for reference)
            
        Returns:
            numpy.ndarray: Processed output relevant to the task
        """
        if self.task_type == 'reconstruction':
            # Return the reconstructed image
            if 'reconstruction' in model_output:
                reconstruction = model_output['reconstruction'].squeeze().cpu()
                
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                reconstruction = reconstruction * std + mean
                
                # Convert to numpy and ensure values are in [0, 1]
                reconstruction = reconstruction.clamp(0, 1).permute(1, 2, 0).numpy()
                
                # If original image was grayscale, convert back to grayscale
                if original_image is not None and len(original_image.shape) == 2:
                    reconstruction = np.mean(reconstruction, axis=2)
                
                # Resize to match original image if needed
                if original_image is not None and reconstruction.shape[:2] != original_image.shape[:2]:
                    from data.processing.transforms import resize_image
                    reconstruction = resize_image(reconstruction, 
                                                 (original_image.shape[1], original_image.shape[0]),
                                                 preserve_aspect_ratio=True)
                
                return reconstruction
        
        elif self.task_type in ['enhancement', 'classification', 'segmentation']:
            # For other tasks, return processed features
            latent = model_output['latent'].cpu().numpy()
            
            # For visualization in enhancement mode, average the features
            if self.task_type == 'enhancement':
                # Using the class token (first token) as a feature representation
                features = latent[:, 0, :]  # [B, D]
                
                # Project to 2D for visualization
                from sklearn import decomposition
                pca = decomposition.PCA(n_components=2)
                features_2d = pca.fit_transform(features)
                
                # Normalize to [0, 1]
                features_2d = (features_2d - features_2d.min()) / (features_2d.max() - features_2d.min() + 1e-8)
                
                # Reshape to match original image if needed
                if original_image is not None:
                    from data.processing.transforms import resize_image
                    result = resize_image(features_2d, 
                                         (original_image.shape[1], original_image.shape[0]),
                                         preserve_aspect_ratio=False)
                    return result
                return features_2d
                
            # For classification, return the class token features
            return latent[:, 0, :]  # [B, D]
        
        # Default fallback
        return model_output['latent'].cpu().numpy()

# Register the model with the ModelRegistry
ModelRegistry.register("novel_vit_mae_cxr", ViTMAECXR)