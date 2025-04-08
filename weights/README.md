# Model Weights Directory

This directory contains the pre-trained model weights used by the medical image enhancement application.

## Directory Structure

```
weights/
├── foundational/                # Classic, reliable models
│   ├── denoising/              
│   │   ├── dncnn_gray_blind.pth # DnCNN denoising model
│   │   └── metadata.json        # Model information
│   ├── super_resolution/        
│   │   ├── edsr_x2.pt           # EDSR super-resolution model
│   │   └── metadata.json        # Model information
│   └── artifact_removal/        
│       ├── G_ema_ep_82.pth      # U-Net GAN artifact removal model
│       └── metadata.json        # Model information
└── novel/                       # Cutting-edge experimental models
    └── ...                      # Novel models will go here
```

## Model Information

### Foundational Models

#### DnCNN (Denoising)
- **File**: `denoising/dncnn_gray_blind.pth`
- **Purpose**: Removes noise from medical images
- **Architecture**: CNN with residual learning
- **Source**: [KAIR GitHub Repository](https://github.com/cszn/KAIR)
- **Paper**: "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" (Zhang et al., 2017)

#### EDSR (Super-Resolution)
- **File**: `super_resolution/edsr_x2.pt`
- **Purpose**: Enhances resolution of medical images (2x upscaling)
- **Architecture**: Enhanced Deep Residual Network 
- **Source**: [EDSR-PyTorch GitHub Repository](https://github.com/sanghyun-son/EDSR-PyTorch)
- **Paper**: "Enhanced Deep Residual Networks for Single Image Super-Resolution" (Lim et al., 2017)

#### U-Net GAN (Artifact Removal)
- **File**: `artifact_removal/G_ema_ep_82.pth`
- **Purpose**: Removes various artifacts from medical images
- **Architecture**: U-Net based GAN generator
- **Source**: Internal development
- **Handles**: Motion artifacts, stripe artifacts, ring artifacts, and metal artifacts

## Weight Setup

If you need to download or adapt model weights, use the provided scripts:

1. Set up the directory structure:
   ```
   python scripts/setup_weights_dir.py
   ```

2. Adapt model weights if needed:
   ```
   python scripts/adapt_weights.py --dncnn path/to/dncnn.pth --edsr path/to/edsr.pt --unet path/to/unet.pth
   ```

3. Test the models:
   ```
   python scripts/initialize_models.py --full --output-dir output
   ```

## License

The model weights are subject to their original licenses. See the respective source repositories for details.