#!/usr/bin/env python3
"""
Weight adapters for converting between different model weight formats.
These functions help adapt weights from various sources to work with our model implementations.
"""

import torch
import os
import json

def adapt_dncnn_weights(weight_path, output_path=None):
    """
    Adapt DnCNN weights from 'model.X' format to 'dncnn.X' format.
    
    Args:
        weight_path: Path to the original weight file
        output_path: Path to save the adapted weights (if None, uses original path with '_adapted' suffix)
    
    Returns:
        Path to the adapted weights file
    """
    print(f"Adapting DnCNN weights from: {weight_path}")
    
    # Load original weights
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    
    # Create new state dict
    new_state_dict = {}
    
    # Map keys from 'model.X' to 'dncnn.X'
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key.replace('model.', 'dncnn.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Determine output path
    if output_path is None:
        base_path, ext = os.path.splitext(weight_path)
        output_path = f"{base_path}_adapted{ext}"
    
    # Save adapted weights
    torch.save(new_state_dict, output_path)
    print(f"Saved adapted weights to: {output_path}")
    
    # Update metadata if it exists
    metadata_path = os.path.join(os.path.dirname(weight_path), "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update file reference in metadata
        adapted_filename = os.path.basename(output_path)
        metadata['original_file'] = metadata.get('file', '')
        metadata['file'] = adapted_filename
        metadata['adapted'] = True
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated metadata at: {metadata_path}")
    
    return output_path

def analyze_realergan_weights(weight_path):
    """
    Analyze RealESRGAN weights to understand their structure.
    
    Args:
        weight_path: Path to the weight file
    
    Returns:
        Dictionary with analysis information
    """
    print(f"Analyzing RealESRGAN weights from: {weight_path}")
    
    # Load weights
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    
    # Extract params_ema if present
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    
    # Analyze key structure and tensor shapes
    analysis = {
        'keys': list(state_dict.keys()),
        'input_shape': None,
        'output_shape': None,
        'parameter_count': sum(p.numel() for p in state_dict.values()),
        'layer_count': len(state_dict),
    }
    
    # Find input and output shapes
    for key, value in state_dict.items():
        if key == 'conv_first.weight':
            analysis['input_shape'] = value.shape
        elif key == 'conv_last.weight':
            analysis['output_shape'] = value.shape
    
    # Print summary
    print(f"Weight file contains {analysis['layer_count']} layers with {analysis['parameter_count']} parameters")
    if analysis['input_shape']:
        in_channels = analysis['input_shape'][1]
        print(f"Input channels: {in_channels}")
    if analysis['output_shape']:
        out_channels = analysis['output_shape'][0]
        print(f"Output channels: {out_channels}")
    
    return analysis

def create_realergan_x2_adapter_model(weight_path, output_path=None):
    """
    Create a custom RealESRGAN x2 model that adapts to the unusual input channels.
    This creates a new weight file that can be used with a modified model.
    
    Args:
        weight_path: Path to the original weight file
        output_path: Path to save the adapter model config
    
    Returns:
        Path to the adapter model config
    """
    # Analyze weights
    analysis = analyze_realergan_weights(weight_path)
    
    if not analysis['input_shape']:
        print("Could not determine input shape from weights")
        return None
    
    # Get input channels
    in_channels = analysis['input_shape'][1]
    
    # Create adapter config
    adapter_config = {
        "original_file": os.path.basename(weight_path),
        "input_channels": int(in_channels),
        "input_type": "special",
        "adaptation_required": True,
        "adaptation_method": "channel_adapter",
        "notes": "This model requires a special input preprocessing step to convert 3-channel RGB to the expected input format"
    }
    
    # Determine output path
    if output_path is None:
        dir_path = os.path.dirname(weight_path)
        output_path = os.path.join(dir_path, "realergan_x2_adapter_config.json")
    
    # Save adapter config
    with open(output_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"Created adapter config at: {output_path}")
    
    # Update metadata if it exists
    metadata_path = os.path.join(os.path.dirname(weight_path), "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update metadata with adapter info
        metadata['requires_adapter'] = True
        metadata['adapter_config'] = os.path.basename(output_path)
        metadata['input_channels'] = int(in_channels)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated metadata at: {metadata_path}")
    
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Adapt model weights to work with our implementations")
    parser.add_argument("--dncnn", type=str, help="Path to DnCNN weights to adapt")
    parser.add_argument("--realergan-x2", type=str, help="Path to RealESRGAN x2 weights to analyze and create adapter for")
    parser.add_argument("--output", type=str, default=None, help="Path to save adapted weights (optional)")
    args = parser.parse_args()
    
    if args.dncnn:
        adapt_dncnn_weights(args.dncnn, args.output)
    
    if args.realergan_x2:
        create_realergan_x2_adapter_model(args.realergan_x2, args.output)

if __name__ == "__main__":
    main()