# Model Weights Management

This directory contains the weights for AI models used in the medical image enhancement application.

## Directory Structure

```
model_weights/
├── foundational/                    # Classic, stable models
│   ├── denoising/                   # DnCNN and variants
│   │   ├── dncnn_gray_blind.pth     # Main denoising model
│   │   └── metadata.json            # Model information and usage details
│   ├── super_resolution/            # RealESRGAN and variants
│   │   ├── RealESRGAN_x2.pth        # 2x upscaling model
│   │   ├── RealESRGAN_x4.pth        # 4x upscaling model
│   │   ├── RealESRGAN_x8.pth        # 8x upscaling model
│   │   └── metadata.json            # Model information and usage details
│   └── artifact_removal/            # U-Net GAN and variants
│       ├── G_ema_ep_82.pth          # Generator model
│       └── metadata.json            # Model information and usage details
├── novel/                           # Cutting-edge, experimental models
│   ├── diffusion/                   # Diffusion models
│   ├── transformer/                 # ViT and SWIN transformer models
│   └── stylegan/                    # StyleGAN models for image synthesis
└── utils/                           # Testing and integration utilities
    ├── test_all_models.py           # Main testing script
    ├── test_denoising.py            # Detailed DnCNN test script
    ├── test_super_resolution.py     # Detailed RealESRGAN test script
    ├── test_artifact_removal.py     # Detailed U-Net GAN test script
    └── requirements.txt             # Required packages
```

## Model Information

### Foundational Models

#### DnCNN (Denoising)
- **File**: `dncnn_gray_blind.pth`
- **Source**: KAIR GitHub repository
- **Description**: Deep convolutional neural network for image denoising, using residual learning.
- **Paper**: "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" (Zhang et al., 2017)

#### RealESRGAN (Super-Resolution)
- **Files**: `RealESRGAN_x2.pth`, `RealESRGAN_x4.pth`, `RealESRGAN_x8.pth`
- **Source**: RealESRGAN GitHub repository
- **Description**: Enhanced SRGAN for photo-realistic image super-resolution.
- **Paper**: "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data" (Wang et al., 2021)

#### U-Net GAN (Artifact Removal)
- **File**: `G_ema_ep_82.pth` (Generator model)
- **Description**: GAN-based model using U-Net architecture for removing artifacts from medical images.
- **Note**: Only the generator model is needed for inference.

### Novel Models
*To be populated as we test and integrate these models*

## Usage

### Testing Model Weights

Before integrating models into the main application, it's important to verify that the weights are compatible with our model implementations. The testing scripts in the `utils/` directory can help with this.

#### Testing All Models

```bash
cd model_weights
python utils/test_all_models.py --all
```

#### Testing a Specific Model

```bash
# Test a specific model weight file
python utils/test_all_models.py --weights foundational/denoising/dncnn_gray_blind.pth

# Test a specific denoising model with detailed analysis
python utils/test_denoising.py --weights foundational/denoising/dncnn_gray_blind.pth
```

### Integrating with Main Application

After testing, models can be integrated into the main application using the integration script:

```bash
# Integrate all models
python integration_template.py --all

# Integrate a specific model
python integration_template.py --model denoising --category foundational
```

## Adding New Models

To add a new model:

1. Create a new directory under the appropriate category (foundational or novel)
2. Place the weight file in the directory
3. Create a `metadata.json` file with model information (see existing files for template)
4. Test the model with the appropriate testing script
5. Integrate the model with the main application

## Metadata Format

Each model should have a `metadata.json` file with the following structure:

```json
{
    "model_name": "ModelName",
    "file": "weight_file.pth",
    "task": "task_name",
    "source": {
        "url": "https://example.com/repo",
        "commit": "commit_hash_if_applicable",
        "download_date": "YYYY-MM-DD"
    },
    "paper": {
        "title": "Paper Title",
        "authors": "Author Names",
        "year": 2023,
        "doi": "DOI_if_available"
    },
    "model_info": {
        "architecture": "Description",
        "input_channels": 3,
        "output_channels": 3,
        "depth": 0,
        "parameters": 0
    },
    "usage": {
        "input_format": {
            "channels": 3,
            "normalization": [0, 1],
            "preprocessing": "Description"
        },
        "output_format": {
            "type": "Description",
            "postprocessing": "Description"
        }
    },
    "performance": {
        "metric1": 0,
        "metric2": 0
    },
    "version": "1.0",
    "license": "License Name"
}
```

## License

Please refer to the original licenses of each model. The testing and integration scripts in this repository are licensed under the MIT License.