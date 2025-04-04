#!/usr/bin/env python3
"""
Test script specifically for U-Net GAN model weights.
This script runs a thorough test on the artifact removal model.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

class UNetGAN(nn.Module):
    """U-Net based GAN model for artifact removal"""
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGAN, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Decoder with skip connections
        d1 = self.dec1(e5)
        d2 = self.dec2(torch.cat([d1, e4], 1))
        d3 = self.dec3(torch.cat([d2, e3], 1))
        d4 = self.dec4(torch.cat([d3, e2], 1))
        d5 = self.dec5(torch.cat([d4, e1], 1))
        
        return d5


def load_image(image_path=None, size=(256, 256)):
    """Load an image or create a test image"""
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)
    else:
        print("Creating synthetic test image...")
        # Create a synthetic image with patterns
        img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # Create sample patterns
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        
        # Create a medical image-like pattern (brain-like structure)
        circle1 = ((xx - 0.5)**2 + (yy - 0.5)**2) < 0.15
        circle2 = ((xx - 0.5)**2 + (yy - 0.5)**2) < 0.12
        for i in range(3):
            img_array[:, :, i] = (circle1 & ~circle2) * 200
        
        # Add some additional structures
        folds = np.sin(xx * 8 * np.pi) * np.sin(yy * 8 * np.pi) > 0.3
        img_array[:, :, 0] += folds * 50
        img_array[:, :, 1] += folds * 30
        img_array[:, :, 2] += folds * 40
        
        img = Image.fromarray(img_array)
    
    return img


def add_artifacts(img_tensor, artifact_type='random'):
    """Add different types of artifacts to the image tensor"""
    clean_tensor = img_tensor.clone()
    
    # Image shape
    _, c, h, w = img_tensor.shape
    
    if artifact_type == 'motion':
        # Simulate motion artifacts (blurring in one direction)
        kernel_size = 15
        kernel = torch.zeros((c, 1, kernel_size, 1), device=img_tensor.device)
        kernel[:, 0, :, 0] = 1.0 / kernel_size
        
        # Apply convolution for horizontal motion blur
        with torch.no_grad():
            img_tensor = torch.nn.functional.conv2d(
                img_tensor, kernel, padding=(kernel_size//2, 0), groups=c
            )
    
    elif artifact_type == 'stripe':
        # Add horizontal stripe artifacts
        stripe_mask = torch.zeros_like(img_tensor)
        for i in range(0, h, 8):
            stripe_mask[:, :, i:i+3, :] = 1.0
        
        # Apply stripes to image
        noise = torch.randn_like(img_tensor) * 0.2
        img_tensor = img_tensor * (1 - stripe_mask) + stripe_mask * noise
    
    elif artifact_type == 'missing':
        # Simulate missing regions
        mask = torch.ones_like(img_tensor)
        # Create a rectangular missing region
        mask[:, :, h//4:3*h//4, w//4:3*w//4] = 0
        
        # Apply mask to image
        img_tensor = img_tensor * mask
    
    else:  # random artifacts
        # Add random noise patterns
        noise_mask = torch.rand_like(img_tensor) > 0.9
        noise = torch.randn_like(img_tensor) * 0.3
        img_tensor = img_tensor * (~noise_mask) + noise * noise_mask
    
    # Ensure values are in valid range
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    return clean_tensor, img_tensor


def test_unet_gan(weight_path, test_image=None, artifact_types=None):
    """Test U-Net GAN model with different artifact types"""
    if artifact_types is None:
        artifact_types = ['motion', 'stripe', 'missing', 'random']
    
    print(f"Testing U-Net GAN with weights from: {weight_path}")
    
    # Create model
    model = UNetGAN()
    
    # Load pretrained weights
    try:
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        # Handle different state dict formats
        if isinstance(state_dict, dict) and 'G' in state_dict:
            state_dict = state_dict['G']
        # Allow missing or unexpected keys
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return False
    
    # Set model to evaluation mode
    model.eval()
    
    # Load or create test image
    img = load_image(test_image)
    
    # Convert to tensor [1, 3, H, W]
    clean_np = np.array(img).astype(np.float32) / 255.0
    clean_tensor = torch.from_numpy(clean_np.transpose(2, 0, 1)).unsqueeze(0)  # Add batch dimension
    
    # Create a figure for results
    num_artifacts = len(artifact_types)
    fig, axes = plt.subplots(num_artifacts, 3, figsize=(15, 5 * num_artifacts))
    
    results = []
    
    # Test with different artifact types
    for i, artifact_type in enumerate(artifact_types):
        print(f"\nTesting with artifact type: {artifact_type}")
        
        # Add artifacts to the clean image
        clean_tensor_copy, artifacted_tensor = add_artifacts(clean_tensor, artifact_type)
        
        # Run inference
        with torch.no_grad():
            try:
                deartifacted_tensor = model(artifacted_tensor)
                
                # Convert tensors to numpy arrays
                clean_np = clean_tensor_copy.squeeze().cpu().numpy().transpose(1, 2, 0)
                artifacted_np = artifacted_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
                deartifacted_np = deartifacted_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
                
                # Clip to valid range
                clean_np = np.clip(clean_np, 0, 1)
                artifacted_np = np.clip(artifacted_np, 0, 1)
                deartifacted_np = np.clip(deartifacted_np, 0, 1)
                
                # Calculate metrics
                psnr_artifacted = peak_signal_noise_ratio(clean_np, artifacted_np, data_range=1.0)
                ssim_artifacted = structural_similarity(clean_np, artifacted_np, data_range=1.0, multichannel=True)
                
                psnr_deartifacted = peak_signal_noise_ratio(clean_np, deartifacted_np, data_range=1.0)
                ssim_deartifacted = structural_similarity(clean_np, deartifacted_np, data_range=1.0, multichannel=True)
                
                print(f"Artifacted image: PSNR = {psnr_artifacted:.2f}dB, SSIM = {ssim_artifacted:.4f}")
                print(f"Deartifacted image: PSNR = {psnr_deartifacted:.2f}dB, SSIM = {ssim_deartifacted:.4f}")
                print(f"Improvement: PSNR = {psnr_deartifacted - psnr_artifacted:.2f}dB, SSIM = {ssim_deartifacted - ssim_artifacted:.4f}")
                
                results.append({
                    'artifact_type': artifact_type,
                    'psnr_artifacted': float(psnr_artifacted),
                    'ssim_artifacted': float(ssim_artifacted),
                    'psnr_deartifacted': float(psnr_deartifacted),
                    'ssim_deartifacted': float(ssim_deartifacted),
                    'psnr_improvement': float(psnr_deartifacted - psnr_artifacted),
                    'ssim_improvement': float(ssim_deartifacted - ssim_artifacted)
                })
                
                # Plot results
                axes[i, 0].imshow(clean_np)
                axes[i, 0].set_title(f'Clean Image')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(artifacted_np)
                axes[i, 1].set_title(f'Artifacted Image ({artifact_type})\nPSNR: {psnr_artifacted:.2f}dB, SSIM: {ssim_artifacted:.4f}')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(deartifacted_np)
                axes[i, 2].set_title(f'Deartifacted Image\nPSNR: {psnr_deartifacted:.2f}dB, SSIM: {ssim_deartifacted:.4f}')
                axes[i, 2].axis('off')
                
            except Exception as e:
                print(f"❌ Inference failed: {e}")
                import traceback
                traceback.print_exc()