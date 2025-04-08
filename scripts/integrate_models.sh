#!/bin/bash
# Script to integrate pre-trained model weights into the application

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Create required directories
echo "Setting up directory structure..."
python "$SCRIPT_DIR/setup_weights_dir.py"

# Check for available weight files in a standard location
WEIGHTS_DIR="$ROOT_DIR/model_weights"
echo "Checking for weight files in $WEIGHTS_DIR..."

DNCNN_ARGS=""
if [ -f "$WEIGHTS_DIR/foundational/denoising/dncnn_gray_blind.pth" ]; then
    DNCNN_ARGS="--dncnn $WEIGHTS_DIR/foundational/denoising/dncnn_gray_blind.pth"
    echo "Found DnCNN weights"
fi

EDSR_ARGS=""
if [ -f "$WEIGHTS_DIR/foundational/super_resolution/edsr_x2.pt" ]; then
    EDSR_ARGS="--edsr $WEIGHTS_DIR/foundational/super_resolution/edsr_x2.pt"
    echo "Found EDSR weights"
fi

REALSR_ARGS=""
if [ -f "$WEIGHTS_DIR/foundational/super_resolution/RealESRGAN_x4.pth" ]; then
    REALSR_ARGS="--realsr $WEIGHTS_DIR/foundational/super_resolution/RealESRGAN_x4.pth"
    echo "Found RealESRGAN weights"
fi

UNET_ARGS=""
if [ -f "$WEIGHTS_DIR/foundational/artifact_removal/G_ema_ep_82.pth" ]; then
    UNET_ARGS="--unet $WEIGHTS_DIR/foundational/artifact_removal/G_ema_ep_82.pth"
    echo "Found U-Net GAN weights"
fi

# Check if a test image is provided
IMAGE_ARGS=""
if [ "$1" != "" ] && [ -f "$1" ]; then
    IMAGE_ARGS="--image $1"
    echo "Using test image: $1"
fi

# Run the integration script
echo "Running integration script..."
python "$SCRIPT_DIR/integrate_models.py" $DNCNN_ARGS $EDSR_ARGS $REALSR_ARGS $UNET_ARGS $IMAGE_ARGS

# Print success message
echo ""
echo "Integration complete!"
echo "You can now use the foundational models in the application."
echo "Run the application with: python main.py --use-foundational"