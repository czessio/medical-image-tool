#!/usr/bin/env python3
"""
Simple test script for the official DnCNN weights.
This script uses the exact model structure expected by the weights.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

class DnCNNOriginal(nn.Module):
    """Official DnCNN structure that exactly matches the weight file"""
    def __init__(self):
        super(DnCNNOriginal, self).__init__()
        
        # Create a sequential model that will be filled with layers
        # This will match the structure in the weight file exactly
        self.model = nn.Sequential()
        
        # First layer - Conv + ReLU
        self.model.add_module('0', nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True))
        self.model.add_module('1', nn.ReLU(inplace=True))
        
        # Middle layers - Conv + BN + ReLU
        for i in range(2, 32, 2):
            self.model.add_module(f'{i}', nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True))
            self.model.add_module(f'{i+1}', nn.ReLU(inplace=True))
            
        # Last layer - Conv
        self.model.add_module('32', nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True))
    
    def forward(self, x):
        noise = self.model(x)
        return x - noise  # Return the denoised image (input - predicted noise)


def load_image(image_path=None, size=(256, 256)):
    """Load an image or create a test image"""
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(size)
    else:
        print("Creating synthetic test image...")
        # Create a synthetic image with patterns
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        
        # Create various patterns
        circle = ((xx - 0.5)**2 + (yy - 0.5)**2) < 0.1
        lines = np.sin(20 * xx) > 0
        gradient = xx * yy
        
        # Combine patterns
        img_array = circle.astype(np.float32) * 0.5 + lines.astype(np.float32) * 0.25 + gradient * 0.25
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
    
    return img


def add_noise(img_tensor, noise_level=25):
    """Add Gaussian noise to the image tensor"""
    # Convert noise level from [0, 255] to [0, 1]
    sigma = noise_level / 255.0
    noise = torch.randn_like(img_tensor) * sigma
    noisy_img = img_tensor + noise
    noisy_img = torch.clamp(noisy_img, 0, 1)
    return noisy_img


def test_dncnn(weight_path, image_path=None, noise_level=25):
    """Test DnCNN model with the specified noise level"""
    print(f"Testing DnCNN with weights from: {weight_path}")
    
    # Create model
    model = DnCNNOriginal()
    
    # Load pretrained weights
    try:
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return False
    
    # Set model to evaluation mode
    model.eval()
    
    # Load or create test image
    img = load_image(image_path)
    
    # Convert to tensor [1, 1, H, W]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Add noise to the clean image
    noisy_img_tensor = add_noise(img_tensor, noise_level)
    
    # Run inference
    with torch.no_grad():
        try:
            # DnCNN directly outputs the denoised image in our implementation
            denoised_img_tensor = model(noisy_img_tensor)
            denoised_img_tensor = torch.clamp(denoised_img_tensor, 0, 1)
            
            # Convert tensors to numpy arrays
            clean_np = img_tensor.squeeze().numpy()
            noisy_np = noisy_img_tensor.squeeze().numpy()
            denoised_np = denoised_img_tensor.squeeze().numpy()
            
            # Create results directory
            result_dir = os.path.dirname(weight_path)
            os.makedirs(result_dir, exist_ok=True)
            
            # Create a figure for results
            plt.figure(figsize=(15, 5))
            
            # Plot clean image
            plt.subplot(1, 3, 1)
            plt.imshow(clean_np, cmap='gray')
            plt.title('Clean Image')
            plt.axis('off')
            
            # Plot noisy image
            plt.subplot(1, 3, 2)
            plt.imshow(noisy_np, cmap='gray')
            plt.title(f'Noisy Image (σ={noise_level})')
            plt.axis('off')
            
            # Plot denoised image
            plt.subplot(1, 3, 3)
            plt.imshow(denoised_np, cmap='gray')
            plt.title('Denoised Image')
            plt.axis('off')
            
            # Save the figure
            result_path = os.path.join(result_dir, f"denoising_result_{noise_level}.png")
            plt.tight_layout()
            plt.savefig(result_path)
            plt.close()
            print(f"✅ Result saved to: {result_path}")
            
            return True
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DnCNN model")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image", type=str, default=None, help="Path to test image (optional)")
    parser.add_argument("--noise", type=int, default=25, help="Noise level (default: 25)")
    args = parser.parse_args()
    
    test_dncnn(args.weights, args.image, args.noise)


if __name__ == "__main__":
    main()