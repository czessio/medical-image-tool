import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_files():
    """Check for model files in expected locations and suggest fixes."""
    # Define expected model paths
    expected_paths = [
        "weights/foundational/denoising/dncnn_gray_blind.pth",
        "weights/foundational/denoising/dncnn_25.pth",
        "weights/foundational/super_resolution/edsr_x2.pt",
        "weights/foundational/super_resolution/RealESRGAN_x2.pth",
        "weights/foundational/artifact_removal/G_ema_ep_82.pth"
    ]
    
    # Find all weight files
    found_files = []
    for root, _, files in os.walk("weights"):
        for file in files:
            if file.endswith((".pth", ".pt")) and not file.endswith((".download", ".tmp")):
                found_files.append(os.path.join(root, file))
    
    # Check expected paths
    print("\nChecking expected model paths:")
    for path in expected_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")
    
    # Show all found model files
    print("\nAll found model files:")
    for found in found_files:
        if found not in expected_paths:
            print(f"📦 Found: {found}")
    
    # Check config file
    try:
        from utils.config import Config
        config = Config()
        
        print("\nChecking model paths in config:")
        
        model_types = ["denoising", "super_resolution", "artifact_removal"]
        model_names = ["dncnn_denoiser", "edsr_super_resolution", "unet_artifact_removal"]
        
        for model_type, model_name in zip(model_types, model_names):
            path = config.get(f"models.{model_type}.foundational.{model_name}.model_path")
            print(f"Config path for {model_name}: {path}")
            
            if path and "foundational/foundational" in path:
                print(f"   ⚠️ Path has duplicated 'foundational' directory")
                
            if path and not os.path.exists(path):
                print(f"   ❌ File does not exist at this path")
                # Check for file with different name in same directory
                dir_path = os.path.dirname(path)
                if os.path.exists(dir_path):
                    files = os.listdir(dir_path)
                    if files:
                        print(f"   💡 Found files in same directory: {', '.join(files)}")
    except Exception as e:
        print(f"Error checking config: {e}")
    
    # Suggest fixes
    print("\nSuggested fixes:")
    print("1. Run the following command to update your config:")
    print("   python -c \"from utils.config import Config; config = Config(); " +
          "config.set('models.use_novel', False); " +
          "config.set('models.denoising.foundational.dncnn_denoiser.model_path', 'weights/foundational/denoising/dncnn_gray_blind.pth'); " +
          "config.set('models.super_resolution.foundational.edsr_super_resolution.model_path', 'weights/foundational/super_resolution/edsr_x2.pt'); " +
          "config.set('models.artifact_removal.foundational.unet_artifact_removal.model_path', 'weights/foundational/artifact_removal/G_ema_ep_82.pth'); " +
          "config.save()\"")
    print("2. Make sure your model files have the correct names")
    print("3. Check the config.json file at ~/.medimage_enhancer/config.json")

if __name__ == "__main__":
    check_model_files()