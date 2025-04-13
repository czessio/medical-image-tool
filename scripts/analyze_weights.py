# analyze_weights.py
import torch
import argparse
import pprint

def analyze_weight_file(weight_path):
    """Analyze the structure of a PyTorch weight file."""
    print(f"Analyzing weight file: {weight_path}")
    
    # Load the state dict
    state_dict = torch.load(weight_path, map_location='cpu')
    
    # Check if it's a model, optimizer, or just a state dict
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            print("Found 'state_dict' key, extracting it")
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            print("Found 'model' key, extracting it")
            state_dict = state_dict['model']
        elif 'params_ema' in state_dict:
            print("Found 'params_ema' key, extracting it")
            state_dict = state_dict['params_ema']
    
    # Analyze the structure
    print("\nState Dict Structure:")
    print(f"Total keys: {len(state_dict)}")
    
    # Look for specific patterns
    input_channels = None
    output_channels = None
    
    # Analyze key dimensions for insights
    print("\nKey Dimensions:")
    for key, tensor in state_dict.items():
        print(f"{key}: {tensor.shape}")
        
        # Look for input/output channels
        if 'first' in key and 'weight' in key and len(tensor.shape) == 4:
            input_channels = tensor.shape[1]
            print(f"  - Possible input channels: {input_channels}")
        
        if 'last' in key and 'weight' in key and len(tensor.shape) == 4:
            output_channels = tensor.shape[0]
            print(f"  - Possible output channels: {output_channels}")
    
    # Print summary
    print("\nSummary:")
    if input_channels is not None:
        print(f"Input channels: {input_channels}")
    if output_channels is not None:
        print(f"Output channels: {output_channels}")
    
    return state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PyTorch weight files")
    parser.add_argument("weight_path", help="Path to the weight file to analyze")
    args = parser.parse_args()
    
    analyze_weight_file(args.weight_path)