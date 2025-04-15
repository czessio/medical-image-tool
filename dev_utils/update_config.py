from utils.config import Config

def update_config_for_local_models():
    """Update the config to use the local model files."""
    config = Config()
    
    # Set to use foundational models
    config.set("models.use_novel", False)
    
    # Update paths to match the actual files you have
    config.set("models.denoising.foundational.dncnn_denoiser.model_path", 
               "weights/foundational/denoising/dncnn_25.pth")
    
    config.set("models.super_resolution.foundational.edsr_super_resolution.model_path", 
               "weights/foundational/super_resolution/RealESRGAN_x2.pth")
    
    config.set("models.artifact_removal.foundational.unet_artifact_removal.model_path", 
               "weights/foundational/artifact_removal/G_ema_ep_82.pth")
    
    # Save the updated config
    config.save()
    
    print("Configuration updated to use your local model files:")
    print(f"Denoising: dncnn_25.pth")
    print(f"Super-resolution: RealESRGAN_x2.pth")
    print(f"Artifact removal: G_ema_ep_82.pth")

if __name__ == "__main__":
    update_config_for_local_models()