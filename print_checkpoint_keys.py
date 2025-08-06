import os
import torch
import argparse
from collections import OrderedDict

def print_checkpoint_structure(checkpoint_path):
    """Print the structure of keys in a PyTorch checkpoint."""
    print(f"Examining checkpoint at: {checkpoint_path}")
    
    # Check if it's a directory (accelerate save_state) or a file
    if os.path.isdir(checkpoint_path):
        # Look for pytorch_model.bin or model.safetensors
        model_files = [
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            os.path.join(checkpoint_path, "model.safetensors")
        ]
        
        # Also check for accelerator state files
        accelerator_files = [f for f in os.listdir(checkpoint_path) 
                            if f.startswith("pytorch_model_") and f.endswith(".bin")]
        
        if accelerator_files:
            print(f"Found {len(accelerator_files)} accelerator state files:")
            for file in accelerator_files:
                file_path = os.path.join(checkpoint_path, file)
                print(f"\nExamining accelerator file: {file}")
                try:
                    # Use weights_only=False for compatibility
                    state_dict = torch.load(file_path, map_location="cpu", weights_only=False)
                    print_state_dict_structure(state_dict)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"\nExamining model file: {os.path.basename(model_file)}")
                try:
                    if model_file.endswith('.safetensors'):
                        # For safetensors files, use the safetensors library
                        try:
                            from safetensors import safe_open
                            with safe_open(model_file, framework="pt") as f:
                                # Get all tensor names
                                tensor_names = list(f.keys())
                                print(f"Safetensors file contains {len(tensor_names)} keys")
                                
                                # Check for register-related keys
                                register_keys = [k for k in tensor_names if 'register' in k.lower()]
                                if register_keys:
                                    print("\nRegister-related keys:")
                                    for k in register_keys:
                                        print(f"  {k}")
                                
                                # Print first 10 keys as sample
                                print("\nSample keys (first 10):")
                                for i, k in enumerate(tensor_names[:10]):
                                    print(f"  {k}")
                                
                                # Check for vision model keys
                                vision_keys = [k for k in tensor_names if 'vision_model' in k]
                                if vision_keys:
                                    print(f"\nFound {len(vision_keys)} vision model keys")
                                    print("Sample vision model keys (first 5):")
                                    for k in vision_keys[:5]:
                                        print(f"  {k}")
                        except ImportError:
                            print("safetensors library not installed. Run 'pip install safetensors' to examine safetensors files.")
                        except Exception as e:
                            print(f"Error examining safetensors file: {e}")
                    else:
                        # For regular PyTorch files
                        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
                        print_state_dict_structure(state_dict)
                except Exception as e:
                    print(f"Error loading {model_file}: {e}")
                    print("Trying with weights_only=False...")
                    try:
                        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
                        print_state_dict_structure(state_dict)
                    except Exception as e2:
                        print(f"Still failed with weights_only=False: {e2}")
                break
        else:
            print("No standard model file found in directory.")
            
        # Check for config.json
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            import json
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print("\nConfig contains the following keys:")
                print(list(config.keys()))
                
                # Check for num_registers in config
                if "num_registers" in config:
                    print(f"num_registers in config: {config['num_registers']}")
                else:
                    print("num_registers not found in config")
                    
                # Check vision_config
                if "vision_config" in config and isinstance(config["vision_config"], dict):
                    if "num_registers" in config["vision_config"]:
                        print(f"vision_config.num_registers: {config['vision_config']['num_registers']}")
            except Exception as e:
                print(f"Error reading config.json: {e}")
    else:
        # Direct checkpoint file
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print_state_dict_structure(state_dict)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

def print_state_dict_structure(state_dict):
    """Print the structure of a state dict."""
    if isinstance(state_dict, OrderedDict) or isinstance(state_dict, dict):
        # This is a standard state dict
        print(f"State dict contains {len(state_dict)} keys")
        
        # Check for register-related keys
        register_keys = [k for k in state_dict.keys() if 'register' in k.lower()]
        if register_keys:
            print("\nRegister-related keys:")
            for k in register_keys:
                shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'N/A'
                print(f"  {k}: {shape}")
        
        # Print first 10 keys as sample
        print("\nSample keys (first 10):")
        for i, (k, v) in enumerate(state_dict.items()):
            if i >= 10:
                break
            shape = v.shape if hasattr(v, 'shape') else 'N/A'
            print(f"  {k}: {shape}")
            
        # Check for vision model keys
        vision_keys = [k for k in state_dict.keys() if 'vision_model' in k]
        if vision_keys:
            print(f"\nFound {len(vision_keys)} vision model keys")
            print("Sample vision model keys (first 5):")
            for k in vision_keys[:5]:
                shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'N/A'
                print(f"  {k}: {shape}")
    else:
        # This might be an accelerator state or something else
        print(f"State dict is not a standard OrderedDict/dict. Type: {type(state_dict)}")
        if hasattr(state_dict, 'keys'):
            print(f"Keys: {list(state_dict.keys())}")
        else:
            print("Cannot extract keys from this object")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the structure of a PyTorch checkpoint")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file or directory")
    args = parser.parse_args()
    
    print_checkpoint_structure(args.checkpoint_path)
