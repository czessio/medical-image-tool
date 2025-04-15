import os
import shutil
from pathlib import Path

def clean_weights_directory():
    weights_dir = Path("weights")
    
    # Check if we have the duplicate structure
    nested_foundational = weights_dir / "foundational" / "foundational"
    if nested_foundational.exists():
        print("Found nested foundational directory, cleaning up...")
        
        # Copy all files from nested directories to their parent directories
        for model_type in ["denoising", "super_resolution", "artifact_removal"]:
            nested_dir = nested_foundational / model_type
            if nested_dir.exists():
                parent_dir = weights_dir / "foundational" / model_type
                parent_dir.mkdir(exist_ok=True)
                
                for file in nested_dir.glob("*"):
                    if file.is_file():
                        target_path = parent_dir / file.name
                        if not target_path.exists():
                            print(f"Moving {file} to {target_path}")
                            shutil.copy2(file, target_path)
        
        # Remove the nested foundational directory
        print(f"Removing nested directory: {nested_foundational}")
        shutil.rmtree(nested_foundational)
        
    print("Weights directory structure cleaned")

# Run the cleanup
clean_weights_directory()