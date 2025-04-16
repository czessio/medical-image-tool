"""
Model quantization utilities for the medical image enhancement application.
Provides functions to quantize PyTorch models for faster inference.
"""
import logging
import os
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.quantization
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelQuantizer:
    """
    Quantizes PyTorch models to reduce memory usage and improve inference speed.
    """
    
    @staticmethod
    def quantize_model(model, quantization_type="dynamic", dtype="int8"):
        """
        Quantize a PyTorch model to reduce memory usage and improve inference speed.
        
        Args:
            model: PyTorch model to quantize
            quantization_type: Type of quantization to apply ('dynamic', 'static', or 'qat')
            dtype: Data type for quantization ('int8' or 'fp16')
            
        Returns:
            torch.nn.Module: Quantized model
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is not available")
            return model
        
        # Save the original device
        original_device = next(model.parameters()).device
        
        try:
            # Move model to CPU for quantization
            model = model.cpu()
            
            # Clone the model if possible to avoid modifying the original
            try:
                import copy
                model_for_quantization = copy.deepcopy(model)
            except:
                logger.warning("Could not clone model, quantizing original model")
                model_for_quantization = model
            
            # Make sure model is in eval mode
            model_for_quantization.eval()
            
            # Apply different quantization based on type and dtype
            if dtype == "fp16" and torch.cuda.is_available():
                # FP16 quantization (half-precision)
                logger.info("Applying FP16 quantization")
                quantized_model = model_for_quantization.half()
            elif quantization_type == "dynamic":
                # Dynamic quantization
                logger.info("Applying dynamic INT8 quantization")
                
                # Prepare for dynamic quantization
                quantization_config = torch.quantization.get_default_qconfig("fbgemm")
                
                # Fuse modules for better quantization results if possible
                model_for_quantization = ModelQuantizer._fuse_modules(model_for_quantization)
                
                # Apply dynamic quantization
                torch.quantization.prepare(model_for_quantization, inplace=True)
                quantized_model = torch.quantization.convert(model_for_quantization, inplace=False)
                
            elif quantization_type == "static":
                # Static quantization (requires calibration data which we don't have here)
                logger.info("Applying static INT8 quantization (without calibration)")
                
                # Add observer for activations
                model_for_quantization.qconfig = torch.quantization.get_default_qconfig("fbgemm")
                
                # Fuse modules for better quantization results if possible
                model_for_quantization = ModelQuantizer._fuse_modules(model_for_quantization)
                
                # Prepare and convert
                model_for_quantization = torch.quantization.prepare(model_for_quantization)
                quantized_model = torch.quantization.convert(model_for_quantization)
                
            else:
                logger.warning(f"Unsupported quantization type: {quantization_type}, falling back to FP32")
                quantized_model = model_for_quantization
            
            # Move the model back to the original device
            quantized_model = quantized_model.to(original_device)
            
            # Calculate and log model size reduction
            original_size = ModelQuantizer._get_model_size(model)
            quantized_size = ModelQuantizer._get_model_size(quantized_model)
            
            logger.info(f"Model quantization complete. Size reduced from {original_size:.2f}MB to {quantized_size:.2f}MB ({(1 - quantized_size/original_size)*100:.1f}% reduction)")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Error during model quantization: {e}")
            
            # Restore original model to its device
            model = model.to(original_device)
            
            return model
    
    @staticmethod
    def _fuse_modules(model):
        """
        Fuse modules like Conv+BN+ReLU for more efficient quantization.
        This is model-specific and will be a no-op for models where fusion isn't applicable.
        
        Args:
            model: PyTorch model
            
        Returns:
            torch.nn.Module: Model with fused modules
        """
        try:
            # List all possible modules to fuse
            modules_to_fuse = []
            
            # Look for Conv+BN+ReLU patterns or Linear+ReLU patterns
            for name, module in model.named_children():
                if isinstance(module, nn.Sequential):
                    for i in range(len(module) - 2):
                        if (isinstance(module[i], nn.Conv2d) and 
                            isinstance(module[i+1], nn.BatchNorm2d) and 
                            isinstance(module[i+2], nn.ReLU)):
                            modules_to_fuse.append([f"{name}.{i}", f"{name}.{i+1}", f"{name}.{i+2}"])
                        
                        if (isinstance(module[i], nn.Linear) and 
                            isinstance(module[i+1], nn.ReLU)):
                            modules_to_fuse.append([f"{name}.{i}", f"{name}.{i+1}"])
            
            # If modules to fuse were found, fuse them
            if modules_to_fuse:
                logger.info(f"Fusing {len(modules_to_fuse)} module groups for quantization")
                model = torch.quantization.fuse_modules(model, modules_to_fuse)
            
            return model
        except Exception as e:
            logger.warning(f"Module fusion failed: {e}")
            return model
    
    @staticmethod
    def _get_model_size(model):
        """
        Calculate the size of a PyTorch model in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            float: Model size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb
    
    @staticmethod
    def is_quantization_supported():
        """
        Check if quantization is supported in the current environment.
        
        Returns:
            bool: True if quantization is supported
        """
        if not TORCH_AVAILABLE:
            return False
        
        try:
            import torch.quantization
            return True
        except ImportError:
            return False

def quantize_model_to_file(model, output_path, quantization_type="dynamic", dtype="int8"):
    """
    Quantize a model and save it to a file.
    
    Args:
        model: PyTorch model to quantize
        output_path: Path to save the quantized model
        quantization_type: Type of quantization to apply
        dtype: Data type for quantization
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Quantize the model
        quantized_model = ModelQuantizer.quantize_model(model, quantization_type, dtype)
        
        # Save the quantized model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(quantized_model.state_dict(), output_path)
        
        logger.info(f"Quantized model saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving quantized model: {e}")
        return False