# Medical Image Enhancement - Model Integration Documentation

This document describes how deep learning models are integrated into the Medical Image Enhancement application. The application uses PyTorch-based models for denoising, super-resolution, and artifact removal in medical images.

## Architecture Overview

The model integration system consists of the following components:

1. **ModelRegistry**: Centralized registry for model classes to allow instantiation by name
2. **ModelDownloader**: Utility for downloading and verifying model weights
3. **ModelManager**: Manager for loading, caching, and inference with models
4. **ModelInitializer**: Helper for initializing models at application startup
5. **InferencePipeline**: Pipeline for executing multiple models in sequence
6. **CleaningPipeline**: Specialized pipeline for image enhancement operations

## Model Organization

Models are organized into two categories:

- **Foundational Models**: Classic, well-established models (DnCNN, EDSR, U-Net)
- **Novel Models**: Cutting-edge models (Diffusion, SwinIR, StyleGAN)

Each model has a consistent interface through the `TorchModel` base class, which handles preprocessing, inference, and postprocessing.

## Adding a New Model

To add a new model to the application:

1. Create a new model class that inherits from `TorchModel`
2. Implement the required methods: `_create_model_architecture`, `preprocess`, `inference`, `postprocess`
3. Register the model with the `ModelRegistry`
4. Add model information to the `MODEL_REGISTRY` in `ModelDownloader`

Example:

```python
from ai.torch_model import TorchModel
from ai.model_registry import ModelRegistry

class MyNewModel(TorchModel):
    def _create_model_architecture(self):
        # Create and return your model architecture
        return model
    
    def preprocess(self, image):
        # Preprocess the input image
        return tensor
    
    def inference(self, preprocessed_tensor):
        # Run inference with the model
        return output
    
    def postprocess(self, model_output, original_image=None):
        # Postprocess the model output
        return result

# Register the model
ModelRegistry.register("my_new_model", MyNewModel)
```

Then add the model information to `ModelDownloader.MODEL_REGISTRY`:

```python
"my_new_model": {
    "url": "https://example.com/my_new_model.pth",
    "md5": "md5_checksum_here",
    "file_name": "my_new_model.pth",
    "description": "My new model for image enhancement",
    "category": "novel", # or "foundational"
    "input_channels": 1,
    "output_channels": 1,
    "grayscale_only": True
}
```

## Model Weights Storage

Model weights are stored in the following directory structure:

```
weights/
├── foundational/
│   ├── dncnn_denoiser.pth
│   ├── edsr_super_resolution.pth
│   └── unet_artifact_removal.pth
└── novel/
    ├── diffusion_denoiser.pth
    ├── swinir_super_resolution.pth
    └── stylegan_artifact_removal.pth
```

The actual paths can be configured in the application settings.

## Model Loading and Caching

The `ModelManager` handles loading models and caching them to avoid reloading the same model multiple times. Models are cached using weak references to allow garbage collection when no longer in use.

The manager also handles automatic downloading of missing models during initialization if enabled in the settings.

## Inference Process

The inference process follows these steps:

1. Input image is preprocessed for the specific model
2. Model runs inference on the preprocessed input
3. Model output is postprocessed to produce the final result
4. Results are passed to the next model in the pipeline if applicable

## Development Guidelines

When implementing or updating models:

1. **Consistent Interface**: All models should follow the same interface pattern
2. **Error Handling**: Robust error handling for model loading and inference failures
3. **Efficient Processing**: Optimize memory usage and processing speed for large images
4. **Proper Validation**: Validate model input and output to prevent unexpected behavior
5. **Documentation**: Document model architecture, parameters, and expected inputs/outputs

## Important File Locations

Here's where to find the key files for model integration:

- **/utils/model_downloader.py**: Model downloading utility
- **/utils/model_manager.py**: Model loading and caching manager
- **/utils/model_initializer.py**: Initialization of models at startup
- **/ai/torch_model.py**: Base class for PyTorch models
- **/ai/model_registry.py**: Registry for model classes
- **/ai/inference_pipeline.py**: Pipeline for sequential model execution
- **/ai/cleaning/inference/cleaning_pipeline.py**: Pipeline for image enhancement
- **/ai/cleaning/models/foundational/**: Implementation of foundational models
- **/ai/cleaning/models/novel/**: Implementation of novel models

## Troubleshooting

Common issues and solutions:

1. **Model loading fails**: Check file paths, model existence, and PyTorch compatibility
2. **CUDA out of memory**: Reduce batch size or image dimensions for processing
3. **Slow inference**: Check device settings, consider using GPU if available
4. **Missing dependencies**: Ensure all required packages are installed

For more detailed information, refer to the implementation of specific models in the source code.