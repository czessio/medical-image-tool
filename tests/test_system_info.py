from pathlib import Path
import sys
import numpy as np

# Add the parent directory to sys.path to make imports work
sys.path.append(str(Path(__file__).parent.parent))
from utils.system_info import get_system_info, get_optimal_device, estimate_image_memory

def test_system_info_contains_required_fields():
    """Test that system info contains all required fields"""
    info = get_system_info()
    
    # Check that main sections exist
    assert "os" in info
    assert "cpu" in info
    assert "memory" in info
    assert "gpu" in info
    
    # Check CPU info
    assert "cores_physical" in info["cpu"]
    assert "cores_logical" in info["cpu"]
    
    # Check memory info
    assert "total_gb" in info["memory"]
    assert info["memory"]["total_gb"] > 0

def test_optimal_device():
    """Test that optimal device is returned"""
    device = get_optimal_device()
    
    # Should be either 'cuda' or 'cpu'
    assert device in ['cuda', 'cpu']

def test_memory_estimation():
    """Test memory estimation for images"""
    # Create a simulated 4K image (3840x2160) with 3 channels
    shape = (2160, 3840, 3)
    
    # Estimate memory (float32)
    mem_gb = estimate_image_memory(shape, dtype=np.float32)
    
    # 4K RGB float32 image should be roughly 0.095 GB
    assert 0.09 < mem_gb < 0.1
    
    # Test with float64 (should be about double)
    mem_gb_float64 = estimate_image_memory(shape, dtype=np.float64)
    assert mem_gb_float64 > mem_gb * 1.9