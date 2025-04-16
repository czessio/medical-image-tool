"""
Batch processing utilities for the medical image enhancement application.
Provides functions to process multiple images in batches for better performance.
"""
import os
import logging
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading

from utils.memory_monitor import memory_monitor, monitor_memory
from utils.system_info import estimate_image_memory

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Processes multiple images in batches for improved performance.
    Optimizes batch size based on available memory and supports multi-threading.
    """
    
    def __init__(self, pipeline, batch_size=None, max_workers=None, use_threading=True):
        """
        Initialize the batch processor.
        
        Args:
            pipeline: The inference pipeline to use for processing
            batch_size: Batch size for processing, or None to auto-determine
            max_workers: Maximum number of worker threads/processes, or None for auto
            use_threading: Whether to use threading (True) or multiprocessing (False)
        """
        self.pipeline = pipeline
        self.batch_size = batch_size
        
        # Determine number of workers
        if max_workers is None:
            # Use number of CPU cores, but cap at 4 to avoid memory issues
            self.max_workers = min(4, multiprocessing.cpu_count())
        else:
            self.max_workers = max_workers
            
        self.use_threading = use_threading
        
        # Start memory monitoring
        memory_monitor.start_monitoring()
    
    def determine_optimal_batch_size(self, sample_image):
        """
        Determine the optimal batch size based on available memory and sample image.
        
        Args:
            sample_image: Sample image to use for memory estimation
            
        Returns:
            int: Optimal batch size
        """
        if self.batch_size is not None:
            return self.batch_size
        
        try:
            # Use the pipeline's built-in estimator if available
            if hasattr(self.pipeline, 'estimate_optimal_batch_size'):
                batch_size = self.pipeline.estimate_optimal_batch_size(sample_image)
            else:
                # Otherwise use the memory monitor's estimator
                batch_size = memory_monitor.estimate_batch_size(sample_image, target_memory_usage=0.7)
            
            # Ensure at least 1 and no more than 16 to avoid excessive memory usage
            batch_size = max(1, min(16, batch_size))
            
            logger.info(f"Using optimal batch size: {batch_size}")
            return batch_size
            
        except Exception as e:
            logger.error(f"Error determining batch size: {e}")
            # Default to conservative batch size of 2
            return 2
    
    @monitor_memory
    def process_images(self, images, output_dir=None, output_prefix="processed_"):
        """
        Process multiple images in batches.
        
        Args:
            images: List of (image_data, image_path) tuples
            output_dir: Directory to save processed images, or None for no saving
            output_prefix: Prefix for output filenames
            
        Returns:
            dict: Dictionary mapping original paths to processed images or output paths
        """
        if not images:
            logger.warning("No images to process")
            return {}
        
        start_time = time.time()
        sample_image = images[0][0] if isinstance(images[0], tuple) else images[0]
        
        # Determine optimal batch size if not specified
        batch_size = self.determine_optimal_batch_size(sample_image)
        
        # Prepare output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract image data and paths
        if isinstance(images[0], tuple):
            # List of (image_data, image_path) tuples
            image_data = [img[0] for img in images]
            image_paths = [img[1] for img in images]
        else:
            # List of image data only
            image_data = images
            image_paths = [f"image_{i}" for i in range(len(images))]
        
        # Calculate total number of batches for progress reporting
        total_batches = (len(image_data) + batch_size - 1) // batch_size
        
        # Process images in batches
        results = {}
        
        if self.max_workers <= 1:
            # Single-threaded processing
            logger.info(f"Processing {len(image_data)} images in {total_batches} batches")
            
            for batch_idx in range(0, len(image_data), batch_size):
                # Get batch
                batch_data = image_data[batch_idx:batch_idx + batch_size]
                batch_paths = image_paths[batch_idx:batch_idx + batch_size]
                
                # Process batch
                logger.info(f"Processing batch {batch_idx//batch_size + 1}/{total_batches}")
                processed_batch = self.pipeline.process_batch(batch_data, batch_size)
                
                # Save results
                for i, (processed, path) in enumerate(zip(processed_batch, batch_paths)):
                    if output_dir:
                        # Save processed image
                        output_path = self._save_processed_image(processed, path, output_dir, output_prefix)
                        results[path] = output_path
                    else:
                        # Just store the processed image
                        results[path] = processed
                
                # Collect garbage after each batch
                memory_monitor.collect_garbage()
        else:
            # Multi-threaded processing
            logger.info(f"Processing {len(image_data)} images in {total_batches} batches using {self.max_workers} workers")
            
            # Create executor
            executor_class = ThreadPoolExecutor if self.use_threading else multiprocessing.Pool
            
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit batch processing tasks
                futures = []
                
                for batch_idx in range(0, len(image_data), batch_size):
                    # Get batch
                    batch_data = image_data[batch_idx:batch_idx + batch_size]
                    batch_paths = image_paths[batch_idx:batch_idx + batch_size]
                    
                    # Submit task
                    if self.use_threading:
                        future = executor.submit(
                            self._process_batch, batch_data, batch_paths, 
                            batch_idx//batch_size + 1, total_batches,
                            output_dir, output_prefix
                        )
                        futures.append(future)
                    else:
                        # For multiprocessing, we need to use apply_async
                        future = executor.apply_async(
                            self._process_batch,
                            (batch_data, batch_paths, batch_idx//batch_size + 1, total_batches, output_dir, output_prefix)
                        )
                        futures.append(future)
                
                # Collect results
                if self.use_threading:
                    for future in as_completed(futures):
                        batch_results = future.result()
                        results.update(batch_results)
                else:
                    # For multiprocessing, we need to get results differently
                    for future in futures:
                        batch_results = future.get()
                        results.update(batch_results)
        
        # Report completion
        elapsed_time = time.time() - start_time
        images_per_second = len(image_data) / elapsed_time
        logger.info(f"Processed {len(image_data)} images in {elapsed_time:.2f} seconds ({images_per_second:.2f} images/sec)")
        
        return results
    
    def _process_batch(self, batch_data, batch_paths, batch_num, total_batches, output_dir, output_prefix):
        """
        Process a single batch of images.
        
        Args:
            batch_data: List of images to process
            batch_paths: List of image paths corresponding to batch_data
            batch_num: Current batch number (for logging)
            total_batches: Total number of batches (for logging)
            output_dir: Directory to save processed images
            output_prefix: Prefix for output filenames
            
        Returns:
            dict: Results for this batch
        """
        try:
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Process the batch
            processed_batch = self.pipeline.process_batch(batch_data)
            
            batch_results = {}
            
            # Save or store results
            for processed, path in zip(processed_batch, batch_paths):
                if output_dir:
                    # Save processed image
                    output_path = self._save_processed_image(processed, path, output_dir, output_prefix)
                    batch_results[path] = output_path
                else:
                    # Just store the processed image
                    batch_results[path] = processed
            
            # Clean up memory
            memory_monitor.collect_garbage()
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            # Return empty results on error
            return {}
    
    def _save_processed_image(self, processed_image, original_path, output_dir, output_prefix):
        """
        Save a processed image to disk.
        
        Args:
            processed_image: Processed image data
            original_path: Path of the original image
            output_dir: Directory to save the processed image
            output_prefix: Prefix for the output filename
            
        Returns:
            str: Path to the saved image
        """
        try:
            from data.io import ImageLoader
            
            # Generate output path
            original_filename = os.path.basename(original_path)
            output_filename = f"{output_prefix}{original_filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the image
            metadata = {"original_path": original_path}
            ImageLoader.save_image(processed_image, output_path, metadata=metadata)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving processed image: {e}")
            return None

def process_directory(pipeline, input_dir, output_dir, file_extensions=None, batch_size=None, max_workers=None):
    """
    Process all images in a directory using batch processing.
    
    Args:
        pipeline: The inference pipeline to use for processing
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        file_extensions: List of file extensions to process, or None for all supported
        batch_size: Batch size for processing, or None to auto-determine
        max_workers: Maximum number of worker threads, or None for auto
        
    Returns:
        dict: Dictionary of results with stats
    """
    from data.io import ImageLoader
    
    # Determine supported file extensions
    if file_extensions is None:
        file_extensions = ImageLoader.get_supported_formats()
    
    # Find all images in the directory
    input_dir = Path(input_dir)
    image_files = []
    
    for ext in file_extensions:
        ext = ext if ext.startswith('.') else f'.{ext}'
        image_files.extend(list(input_dir.glob(f"**/*{ext}")))
    
    if not image_files:
        logger.warning(f"No images found in {input_dir} with extensions {file_extensions}")
        return {"success": False, "processed_count": 0, "error": "No images found"}
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Load and process images
    images = []
    for image_file in image_files:
        try:
            image_data, metadata, is_medical = ImageLoader.load_image(str(image_file))
            images.append((image_data, str(image_file)))
        except Exception as e:
            logger.error(f"Error loading image {image_file}: {e}")
    
    if not images:
        logger.warning("No images could be loaded")
        return {"success": False, "processed_count": 0, "error": "No images could be loaded"}
    
    # Create batch processor
    processor = BatchProcessor(pipeline, batch_size=batch_size, max_workers=max_workers)
    
    # Process images
    start_time = time.time()
    results = processor.process_images(images, output_dir=output_dir)
    elapsed_time = time.time() - start_time
    
    # Return results
    return {
        "success": True,
        "processed_count": len(results),
        "elapsed_time": elapsed_time,
        "images_per_second": len(results) / elapsed_time,
        "result_paths": results
    }