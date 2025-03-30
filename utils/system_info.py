"""
System information utilities for the medical image enhancement application.
Provides information about available hardware and capabilities.
"""
import logging
import platform
import os
import sys
import psutil
import numpy as np

# Try importing torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_system_info():
    """
    Gather system information including OS, CPU, RAM, and GPU if available.
    
    Returns:
        dict: System information
    """
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "cpu": {
            "name": platform.processor(),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True)
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        },
        "gpu": {
            "available": False,
            "devices": []
        }
    }
    
    # Get GPU information if torch is available
    if TORCH_AVAILABLE:
        info["gpu"]["available"] = torch.cuda.is_available()
        if info["gpu"]["available"]:
            for i in range(torch.cuda.device_count()):
                device = {
                    "name": torch.cuda.get_device_name(i),
                    "index": i,
                    "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                }
                info["gpu"]["devices"].append(device)
    
    logger.info(f"System info collected: {platform.system()} with {info['cpu']['cores_logical']} logical cores")
    if info["gpu"]["available"]:
        logger.info(f"GPU available: {info['gpu']['devices'][0]['name']}")
    else:
        logger.info("No GPU detected")
        
    return info

def get_optimal_device():
    """
    Determine the best available device for running models.
    
    Returns:
        str: 'cuda' if GPU is available, otherwise 'cpu'
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    return device

def get_memory_usage():
    """
    Get current memory usage of the application.
    
    Returns:
        float: Memory usage in GB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)
    
    return round(memory_gb, 2)

def estimate_image_memory(shape, dtype=np.float32):
    """
    Estimate memory required for an image.
    
    Args:
        shape: Shape of the image (height, width, channels)
        dtype: Data type of the image
    
    Returns:
        float: Estimated memory in GB
    """
    element_size = np.dtype(dtype).itemsize
    total_bytes = np.prod(shape) * element_size
    total_gb = total_bytes / (1024 ** 3)
    
    return round(total_gb, 4)