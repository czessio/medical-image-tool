"""
Memory monitoring utilities for the medical image enhancement application.
Tracks memory usage during model execution and provides memory optimization techniques.
"""
import os
import logging
import threading
import time
import gc
from functools import wraps

import numpy as np
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    Monitors memory usage of the application and provides memory optimization utilities.
    Tracks both system RAM and GPU memory if available.
    """
    
    def __init__(self, log_interval=5.0, warning_threshold=0.85):
        """
        Initialize the memory monitor.
        
        Args:
            log_interval: How often to log memory usage in seconds
            warning_threshold: Memory usage fraction that triggers warnings (0.0-1.0)
        """
        self.log_interval = log_interval
        self.warning_threshold = warning_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.peak_memory = {
            "ram": 0,
            "gpu": 0
        }
        self.gpu_memory_allocated = {}
        self.process = psutil.Process(os.getpid())
        
    def start_monitoring(self):
        """Start background memory monitoring thread."""
        if self.monitoring:
            logger.debug("Memory monitoring already running")
            return
            
        self.monitoring = True
        self.peak_memory = {"ram": 0, "gpu": 0}
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring thread."""
        if not self.monitoring:
            logger.debug("Memory monitoring not running")
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop that periodically checks memory usage."""
        last_log_time = 0
        
        while self.monitoring:
            current_time = time.time()
            
            # Get current memory usage
            ram_usage = self.get_ram_usage()
            gpu_usage = self.get_gpu_usage() if TORCH_AVAILABLE else None
            
            # Update peak values
            self.peak_memory["ram"] = max(self.peak_memory["ram"], ram_usage["used_gb"])
            if gpu_usage and "used_gb" in gpu_usage:
                self.peak_memory["gpu"] = max(self.peak_memory["gpu"], gpu_usage["used_gb"])
            
            # Check if we should log based on interval
            if current_time - last_log_time >= self.log_interval:
                self._log_memory_state(ram_usage, gpu_usage)
                last_log_time = current_time
                
                # Check for high memory usage
                self._check_memory_warnings(ram_usage, gpu_usage)
            
            # Sleep to avoid consuming too much CPU
            time.sleep(0.1)
    
    def _log_memory_state(self, ram_usage, gpu_usage):
        """Log the current memory state."""
        ram_percent = ram_usage["percent"]
        ram_used = ram_usage["used_gb"]
        ram_total = ram_usage["total_gb"]
        
        if gpu_usage:
            gpu_info = f", GPU: {gpu_usage.get('used_gb', 0):.2f}/{gpu_usage.get('total_gb', 0):.2f} GB ({gpu_usage.get('percent', 0):.1f}%)"
        else:
            gpu_info = ""
            
        logger.info(f"Memory usage - RAM: {ram_used:.2f}/{ram_total:.2f} GB ({ram_percent:.1f}%){gpu_info}, Peak RAM: {self.peak_memory['ram']:.2f} GB")
    
    def _check_memory_warnings(self, ram_usage, gpu_usage):
        """Check for high memory usage and issue warnings if needed."""
        # Check RAM usage
        if ram_usage["percent"] / 100.0 > self.warning_threshold:
            logger.warning(f"High RAM usage: {ram_usage['percent']:.1f}% - application may become unstable")
            # Suggest garbage collection
            self.collect_garbage()
        
        # Check GPU usage if available
        if gpu_usage and "percent" in gpu_usage and gpu_usage["percent"] / 100.0 > self.warning_threshold:
            logger.warning(f"High GPU memory usage: {gpu_usage['percent']:.1f}% - may cause CUDA out of memory errors")
            
            # If very high, suggest clearing CUDA cache
            if gpu_usage["percent"] / 100.0 > 0.95 and TORCH_AVAILABLE:
                logger.info("Clearing CUDA cache to free memory")
                torch.cuda.empty_cache()
    
    def get_ram_usage(self):
        """
        Get current RAM usage information.
        
        Returns:
            dict: RAM usage information
        """
        # Get system memory info
        system_memory = psutil.virtual_memory()
        
        # Get process memory info
        process_memory = self.process.memory_info().rss
        
        return {
            "total_gb": system_memory.total / (1024 ** 3),
            "available_gb": system_memory.available / (1024 ** 3),
            "used_gb": process_memory / (1024 ** 3),
            "percent": system_memory.percent
        }
    
    def get_gpu_usage(self):
        """
        Get current GPU memory usage if available.
        
        Returns:
            dict: GPU memory information or None if not available
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        try:
            # Get memory stats for the current device
            device = torch.cuda.current_device()
            
            # Get allocated and reserved memory
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            
            # Get total memory
            total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            
            # Save per-device allocated memory for tracking
            self.gpu_memory_allocated[device] = allocated
            
            return {
                "device": device,
                "device_name": torch.cuda.get_device_name(device),
                "total_gb": total,
                "used_gb": allocated,
                "reserved_gb": reserved,
                "percent": (allocated / total) * 100 if total > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory usage: {e}")
            return None
    
    def collect_garbage(self):
        """Force garbage collection to free unused memory."""
        start_ram = self.get_ram_usage()["used_gb"]
        start_gpu = self.get_gpu_usage()["used_gb"] if TORCH_AVAILABLE and torch.cuda.is_available() else 0
        
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get new memory usage
        end_ram = self.get_ram_usage()["used_gb"]
        end_gpu = self.get_gpu_usage()["used_gb"] if TORCH_AVAILABLE and torch.cuda.is_available() else 0
        
        # Log the memory freed
        ram_freed = start_ram - end_ram
        gpu_freed = start_gpu - end_gpu
        
        if ram_freed > 0 or gpu_freed > 0:
            logger.info(f"Memory freed - RAM: {ram_freed:.2f} GB, GPU: {gpu_freed:.2f} GB")
        else:
            logger.debug("No significant memory freed by garbage collection")
    
    def estimate_batch_size(self, sample_image, target_memory_usage=0.7):
        """
        Estimate optimal batch size based on sample image size and available memory.
        
        Args:
            sample_image: Sample image as numpy array
            target_memory_usage: Target memory usage as fraction of available
            
        Returns:
            int: Estimated optimal batch size
        """
        # Get memory per image
        if isinstance(sample_image, np.ndarray):
            bytes_per_pixel = sample_image.itemsize
            pixels_per_image = np.prod(sample_image.shape)
            bytes_per_image = bytes_per_pixel * pixels_per_image
        else:
            logger.warning("Sample image is not a numpy array, using conservative estimate")
            bytes_per_image = 50 * 1024 * 1024  # Assume 50MB per image
        
        # Get available memory
        ram_usage = self.get_ram_usage()
        available_ram = ram_usage["available_gb"] * target_memory_usage * (1024 ** 3)  # in bytes
        
        # If GPU is available, consider GPU memory too
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_usage = self.get_gpu_usage()
            if gpu_usage:
                available_gpu = (gpu_usage["total_gb"] - gpu_usage["used_gb"]) * target_memory_usage * (1024 ** 3)
                # Use the smaller of RAM and GPU as the constraint
                available_memory = min(available_ram, available_gpu)
            else:
                available_memory = available_ram
        else:
            available_memory = available_ram
        
        # Estimate batch size
        # Multiply by adjustment factor because model processing needs more memory
        memory_safety_factor = 0.5  # Conservative factor to account for model memory usage
        batch_size = int(available_memory / bytes_per_image * memory_safety_factor)
        
        # Ensure at least batch size 1
        batch_size = max(1, batch_size)
        
        logger.info(f"Estimated optimal batch size: {batch_size} for image shape {sample_image.shape}")
        return batch_size

# Memory usage decorator for monitoring function memory usage
def monitor_memory(func):
    """
    Decorator to monitor memory usage of a function.
    
    Usage:
    @monitor_memory
    def my_function():
        ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a temporary memory monitor if no global one exists
        temp_monitor = None
        if not hasattr(monitor_memory, 'global_monitor') or monitor_memory.global_monitor is None:
            temp_monitor = MemoryMonitor()
            
        monitor = getattr(monitor_memory, 'global_monitor', temp_monitor)
        
        # Get memory usage before
        ram_before = monitor.get_ram_usage()["used_gb"]
        gpu_before = monitor.get_gpu_usage()["used_gb"] if TORCH_AVAILABLE and torch.cuda.is_available() else 0
        
        # Run the function
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Get memory usage after
        ram_after = monitor.get_ram_usage()["used_gb"]
        gpu_after = monitor.get_gpu_usage()["used_gb"] if TORCH_AVAILABLE and torch.cuda.is_available() else 0
        
        # Log memory change
        ram_change = ram_after - ram_before
        gpu_change = gpu_after - gpu_before
        
        logger.info(f"Function {func.__name__} - RAM: {ram_change:+.2f} GB, GPU: {gpu_change:+.2f} GB, Time: {elapsed_time:.2f}s")
        
        # Clean up temporary monitor
        if temp_monitor:
            temp_monitor = None
            
        return result
    
    return wrapper

# Global memory monitor instance
memory_monitor = MemoryMonitor()
monitor_memory.global_monitor = memory_monitor