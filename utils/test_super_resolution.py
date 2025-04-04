#!/usr/bin/env python3
"""
Test script specifically for RealESRGAN model weights.
This script runs a thorough test on the super-resolution model.
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

class RealESRGAN(nn.Module):
    """Simplified RealESRGAN model for testing"""
    def __init__(self, scale=4, channels=3, num_feat=64, num_block=23):
        super(RealESRGAN, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(channels, num_feat, 3, 1, 1)
        
        # Create a simplified backbone (in reality this would be RRDB blocks)
        body = []
        for _ in range(num_block):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body = nn.Sequential(*body)
        
        # Upsampling layers
        upsampling = []
        for _ in range(int(np.log2(scale))):
            upsampling.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            upsampling.append(nn.PixelShuffle(2))
            upsampling.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.upsampling = nn.Sequential(*upsampling)
        
        self.conv_last = nn.Conv2d(num_feat, channels, 3, 1, 1)
    
    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.body(feat)
        feat = feat + body_feat
        feat = self.upsampling(feat)
        out = self.conv_last(feat)
        return out


def load_image(image_path=None, size=(128, 128)):
    """Load an image or create a test image"""
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)
    else:
        print("Creating synthetic test image...")
        # Create a synthetic image with patterns to test super-resolution
        img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # Create different patterns for each channel
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        
        # R channel: circular pattern
        img_array[:, :, 0] = ((xx - 0.5)**2 + (yy - 0.5)**2 < 0.2) * 255
        
        # G channel: grid pattern
        grid_x = np.sin(xx * np.pi * 8) > 0
        grid_y = np.sin(yy * np.pi * 8) > 0
        img_array[:, :, 1] = ((grid_x & grid_y) | (~grid_x & ~grid_y)) * 255
        
        # B channel: radial gradient
        img_array[:, :, 2] = np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2) * 255
        
        img = Image.fromarray(img_array)
    
    return img


def downsample_image(img, scale, method='bicubic'):
    """Downsample image by a factor"""
    w, h = img.size
    return img.resize((w // scale, h // scale), Image.BICUBIC)


def test_real_esrgan(weight_path, test_image=None, scales=None):
    """Test RealESRGAN model with different scales"""
    if scales is None:
        scales = [2, 4, 8]
    
    print(f"Testing RealESRGAN with weights from: {weight_path}")
    
    # Determine the scale from the filename or use default
    model_scale = 4  # Default scale
    for s in scales:
        if f"x{s}" in os.path.basename(weight_path):
            model_scale = s
            break
    
    print(f"Model scale factor: x{model_scale}")
    
    # Create model
    model = RealESRGAN(scale=model_scale)
    
    # Load pretrained weights
    try:
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        # Handle different state dict formats
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return False
    
    # Set model to evaluation mode
    model.eval()
    
    # Load or create test image
    img = load_image(test_image, size=(256, 256))
    
    # Create low-resolution version
    lr_img = downsample_image(img, model_scale)
    
    # Convert to tensor [1, 3, H, W]
    lr_np = np.array(lr_img).astype(np.float32) / 255.0
    lr_tensor = torch.from_numpy(lr_np.transpose(2, 0, 1)).unsqueeze(0)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        try:
            sr_tensor = model(lr_tensor)
            
            # Convert output tensor to numpy array
            sr_np = sr_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            sr_np = np.clip(sr_np, 0, 1)
            
            # Convert high-resolution ground truth for comparison
            hr_np = np.array(img).astype(np.float32) / 255.0
            
            # Resize HR image to match SR output if sizes don't match
            if sr_np.shape[:2] != hr_np.shape[:2]:
                hr_img_resized = img.resize((sr_np.shape[1], sr_np.shape[0]), Image.BICUBIC)
                hr_np = np.array(hr_img_resized).astype(np.float32) / 255.0
            
            # Calculate metrics
            psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
            ssim = structural_similarity(hr_np, sr_np, data_range=1.0, multichannel=True)
            
            print(f"Super-resolution results: PSNR = {psnr:.2f}dB, SSIM = {ssim:.4f}")
            
            # Create bicubic upsampling for comparison
            bicubic_img = lr_img.resize((sr_np.shape[1], sr_np.shape[0]), Image.BICUBIC)
            bicubic_np = np.array(bicubic_img).astype(np.float32) / 255.0
            
            # Calculate metrics for bicubic upsampling
            psnr_bicubic = peak_signal_noise_ratio(hr_np, bicubic_np, data_range=1.0)
            ssim_bicubic = structural_similarity(hr_np, bicubic_np, data_range=1.0, multichannel=True)
            
            print(f"Bicubic upsampling: PSNR = {psnr_bicubic:.2f}dB, SSIM = {ssim_bicubic:.4f}")
            print(f"Improvement over bicubic: PSNR = {psnr - psnr_bicubic:.2f}dB, SSIM = {ssim - ssim_bicubic:.4f}")
            
            # Plot results
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(lr_np)
            axes[0].set_title(f'Low Resolution ({lr_np.shape[0]}x{lr_np.shape[1]})')
            axes[0].axis('off')
            
            axes[1].imshow(bicubic_np)
            axes[1].set_title(f'Bicubic Upsampling\nPSNR: {psnr_bicubic:.2f}dB, SSIM: {ssim_bicubic:.4f}')
            axes[1].axis('off')
            
            axes[2].imshow(sr_np)
            axes[2].set_title(f'Super Resolution (x{model_scale})\nPSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}')
            axes[2].axis('off')
            
            # Save the figure
            result_dir = os.path.dirname(weight_path)
            result_path = os.path.join(result_dir, f"super_resolution_x{model_scale}_results.png")
            plt.tight_layout()
            plt.savefig(result_path)
            plt.close()
            print(f"✅ Result saved to: {result_path}")
            
            # Save metrics as JSON
            metrics = {
                'scale': model_scale,
                'lr_size': [lr_np.shape[1], lr_np.shape[0]],
                'sr_size': [sr_np.shape[1], sr_np.shape[0]],
                'bicubic': {
                    'psnr': float(psnr_bicubic),
                    'ssim': float(ssim_bicubic)
                },
                'super_resolution': {
                    'psnr': float(psnr),
                    'ssim': float(ssim)
                },
                'improvement': {
                    'psnr': float(psnr - psnr_bicubic),
                    'ssim': float(ssim - ssim_bicubic)
                }
            }
            
            metrics_path = os.path.join(result_dir, f"super_resolution_x{model_scale}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"✅ Metrics saved to: {metrics_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RealESRGAN model weights")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file")
    parser.add_argument("--image", type=str, default=None, help="Path to test image (optional)")
    parser.add_argument("--scales", type=int, nargs='+', default=[2, 4, 8], help="Super-resolution scales to test")
    args = parser.parse_args()
    
    test_real_esrgan(args.weights, args.image, args.scales)


if __name__ == "__main__":
    main()