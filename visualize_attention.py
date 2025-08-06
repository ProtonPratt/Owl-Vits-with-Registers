import torch
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
import torchvision.transforms.functional as F
from skimage import exposure
import numpy as np
import json
from tqdm import tqdm
import random

# --- Custom Model Imports ---
# Add the path to your custom transformers implementation
import sys
sys.path.append(os.path.abspath("./transformers/src"))
try:
    from transformers import Owlv2ConfigWithRegisters, Owlv2ForObjectDetectionWithRegisters, Owlv2Processor
    # Also import standard classes if needed (assuming checkpoint_small uses them)
    from transformers import Owlv2Config, Owlv2ForObjectDetection
    print("Successfully imported custom and standard model classes")
except ImportError as e:
    print(f"Failed to import necessary model classes: {e}")
    sys.exit(1)

# --- Configuration ---
CHECKPOINT_DIR_REG = "./froken_backbone_reg/checkpoint-epoch-10-best" # Your fine-tuned checkpoint WITH registers
CHECKPOINT_DIR_NO_REG = "./checkpoint_small" # Your checkpoint WITHOUT registers
IMAGE_SOURCE_DIR = "./val2017/" # Directory containing COCO validation images
ANNOTATION_FILE = "./lvis/lvis_v1_val_10cls_filtered.json" # LVIS annotation file (for category names & image IDs)
OUTPUT_DIR_REG = "./attention_maps_output_registers/" # Output directory for model WITH registers
OUTPUT_DIR_NO_REG = "./attention_maps_output_no_registers/" # Output directory for model WITHOUT registers

# Visualization Parameters
NUM_IMAGES_TO_VISUALIZE = 5 # How many images to process
ATTENTION_CMAP = 'jet' # Colormap for attention visualization
CONTRAST_GAMMA = 0.4 # Gamma correction for attention map contrast (adjust as needed)
# -----------------------------

# --- Visualization Function (Adapted) ---
def plot_attention_map(image, attention_vector, patch_size, h_proc, w_proc, output_path, cmap=ATTENTION_CMAP, gamma=CONTRAST_GAMMA):
    # \"\"\"Visualizes attention to image patches, reshapes, resizes, adjusts contrast, and overlays.\"\"\"
    if not patch_size:
        print(f"Skipping visualization for {os.path.basename(output_path)}: Patch size unknown.")
        return

    if attention_vector is None or attention_vector.numel() == 0:
        print(f"Skipping visualization for {os.path.basename(output_path)}: No valid attention vector provided.")
        return

    try:
        # attention_vector should be 1D tensor of attentions for patch tokens
        num_patches = attention_vector.shape[0]

        # Calculate grid size using processed image dimensions
        num_patches_h = h_proc // patch_size
        num_patches_w = w_proc // patch_size

        if num_patches_h * num_patches_w != num_patches:
            print(f"Warning: Calculated patch grid ({num_patches_h}x{num_patches_w}={num_patches_h*num_patches_w}) "
                  f"mismatch with attention vector length ({num_patches}) for {output_path}. Skipping.")
            return

        if num_patches_h <= 0 or num_patches_w <= 0:
             print(f"Warning: Invalid patch grid dimensions ({num_patches_h}x{num_patches_w}) for {output_path}. Skipping.")
             return

        # Reshape attention to 2D grid
        attention_grid = attention_vector.reshape(num_patches_h, num_patches_w)

        # Resize attention grid to original image size
        attention_grid_resized = F.resize(
            attention_grid.unsqueeze(0).unsqueeze(0),
            size=image.size[::-1], # PIL size is (width, height)
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True
        ).squeeze()

        # Adjust contrast using Gamma Correction
        attention_numpy = attention_grid_resized.numpy()
        attention_non_negative = np.maximum(attention_numpy, 0) # Ensure non-negative
        attention_adjusted = exposure.adjust_gamma(attention_non_negative, gamma=gamma)

        # Plot heatmap overlay
        fig, ax = plt.subplots()
        ax.imshow(image)
        im = ax.imshow(attention_adjusted, cmap=cmap, alpha=1)
        ax.axis('off') # Hide axes
        # Optional: Add colorbar
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved attention map to {output_path}")

    except Exception as e:
        import traceback
        print(f"Error visualizing attention for {output_path}: {e}")
        traceback.print_exc()


# --- Main Execution ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Create both output directories
    os.makedirs(OUTPUT_DIR_REG, exist_ok=True)
    os.makedirs(OUTPUT_DIR_NO_REG, exist_ok=True)

    # 1. Load Annotations (for category names and image selection)
    print(f"Loading annotations from: {ANNOTATION_FILE}")
    try:
        with open(ANNOTATION_FILE, 'r') as f:
            lvis_data = json.load(f)
        cat_id_to_name = {cat['id']: cat['name'] for cat in lvis_data.get('categories', [])}
        image_infos = {img['id']: img for img in lvis_data.get('images', [])}
        all_image_ids = list(image_infos.keys())
        if not cat_id_to_name:
             print("Error: Could not load category names from annotation file.")
             return
        if not all_image_ids:
             print("Error: Could not load image IDs from annotation file.")
             return
        print(f"Loaded {len(cat_id_to_name)} categories and info for {len(all_image_ids)} images.")
    except Exception as e:
        print(f"Error loading annotation file: {e}")
        return

    # 2. Load Processor (Done ONCE - Assuming same processor works for both)
    print(f"Loading processor...")
    try:
        # Try loading from the register checkpoint first, assuming it's compatible
        processor = Owlv2Processor.from_pretrained(CHECKPOINT_DIR_REG, trust_remote_code=True)
        print("Processor loaded.")
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    # 3. Select Images to Visualize (Done ONCE)
    image_ids_to_visualize = random.sample(all_image_ids, min(NUM_IMAGES_TO_VISUALIZE, len(all_image_ids)))
    print(f"Selected {len(image_ids_to_visualize)} images for comparison: {image_ids_to_visualize}")

    # --- Define Model Configurations to Run ---
    model_configs_to_run = [
        {
            "name": "with_registers",
            "checkpoint_dir": CHECKPOINT_DIR_REG,
            "config_class": Owlv2ConfigWithRegisters,
            "model_class": Owlv2ForObjectDetectionWithRegisters,
            "output_dir": OUTPUT_DIR_REG,
            "trust_remote_code": True # Assumed needed for register code
        },
        {
            "name": "no_registers",
            "checkpoint_dir": CHECKPOINT_DIR_NO_REG,
            "config_class": Owlv2Config, # Standard Config
            "model_class": Owlv2ForObjectDetection, # Standard Model
            "output_dir": OUTPUT_DIR_NO_REG,
            "trust_remote_code": False # Assumed not needed for standard model
        },
    ]

    # --- Loop Through Model Configurations ---
    for model_config in model_configs_to_run:
        current_checkpoint = model_config["checkpoint_dir"]
        current_output_dir = model_config["output_dir"]
        model_name = model_config["name"]
        ConfigClass = model_config["config_class"]
        ModelClass = model_config["model_class"]
        trust_code = model_config["trust_remote_code"]

        print(f"\n--- Processing Model: {model_name} ({current_checkpoint}) ---")

        # 4. Load Current Model Configuration
        model = None # Ensure model is reset
        config = None
        patch_size = None
        if device == "cuda": torch.cuda.empty_cache() # Clear cache before loading next model
        
        print(f"Loading model and config for {model_name}...")
        try:
            config = ConfigClass.from_pretrained(current_checkpoint, trust_remote_code=trust_code)
            model = ModelClass.from_pretrained(
                current_checkpoint,
                config=config,
                ignore_mismatched_sizes=True, # Keep this, might be needed for both
                trust_remote_code=trust_code
            )
            model.to(device)
            model.eval()
            # Get patch size from the loaded config
            if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'patch_size'):
                 patch_size = config.vision_config.patch_size
            else:
                 print("Warning: Could not determine patch size from config. Visualization might fail.")
                 # Optionally set a default or skip
                 # patch_size = 16 # Example default
            
            print(f"Model {model_name} loaded. Patch size: {patch_size}")
        except Exception as e:
            print(f"Error loading {model_name} model/processor: {e}. Skipping this model.")
            continue # Skip to the next model configuration

        # 5. Process Each Selected Image for the CURRENT model
        text_queries = [list(cat_id_to_name.values())]

        for img_id in tqdm(image_ids_to_visualize, desc=f"Visualizing Attention ({model_name})"):
            img_info = image_infos.get(img_id)
            if not img_info:
                tqdm.write(f"Skipping image ID {img_id}: Info not found in annotations.")
                continue

            image_filename = img_info.get('file_name', f"{img_id:012d}.jpg")
            image_path = os.path.join(IMAGE_SOURCE_DIR, image_filename)

            if not os.path.exists(image_path):
                tqdm.write(f"Skipping {image_filename}: File not found at {image_path}")
                continue
            
            if not patch_size:
                 tqdm.write(f"Skipping {image_filename} for {model_name}: Patch size unknown.")
                 continue

            try:
                # Load image
                original_image = Image.open(image_path).convert("RGB")

                # Prepare inputs for inference
                inputs = processor(text=text_queries, images=original_image, return_tensors="pt").to(device)

                # Get processed image dimensions
                if 'pixel_values' not in inputs or inputs['pixel_values'].shape[0] != 1:
                    tqdm.write(f"Processor failed for {image_filename}. Skipping.")
                    if 'original_image' in locals(): del original_image
                    if 'inputs' in locals(): del inputs
                    if device == "cuda": torch.cuda.empty_cache()
                    continue
                _, _, h_proc, w_proc = inputs['pixel_values'].shape
                num_image_patches = (h_proc // patch_size) * (w_proc // patch_size)

                # Run inference WITH attention output
                last_layer_attention = None
                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_attentions=True,
                        interpolate_pos_encoding=True # Still important
                    )

                # Extract attention
                if hasattr(outputs, 'vision_model_output') and \
                   hasattr(outputs.vision_model_output, 'attentions') and \
                   outputs.vision_model_output.attentions:
                     last_layer_attention = outputs.vision_model_output.attentions[-1][0].cpu()
                else:
                     tqdm.write(f"Could not extract attention maps for {image_filename}. Skipping.")
                     if 'original_image' in locals(): del original_image
                     if 'inputs' in locals(): del inputs
                     if 'outputs' in locals(): del outputs
                     if device == "cuda": torch.cuda.empty_cache()
                     continue

                # Visualize CLS-to-Patch Attention
                seq_len = last_layer_attention.shape[-1]
                expected_vision_seq_len = 1 + num_image_patches

                if seq_len != expected_vision_seq_len:
                    tqdm.write(f"Warning: Vision attention seq length ({seq_len}) != expected ({expected_vision_seq_len}) for {image_filename}. Skipping.")
                    if 'original_image' in locals(): del original_image
                    if 'inputs' in locals(): del inputs
                    if 'outputs' in locals(): del outputs
                    if 'last_layer_attention' in locals(): del last_layer_attention
                    if device == "cuda": torch.cuda.empty_cache()
                    continue

                cls_token_idx = 0
                image_patch_start_idx = 1
                image_patch_end_idx = image_patch_start_idx + num_image_patches

                attention_slice = last_layer_attention[:, cls_token_idx, image_patch_start_idx:image_patch_end_idx]
                mean_head_attention_to_patches = attention_slice.mean(dim=0).float()

                # Plot and save to the CURRENT model's output directory
                base_filename = os.path.splitext(image_filename)[0]
                attention_output_filename = os.path.join(current_output_dir, f"{base_filename}_attn_cls_to_patches.jpg") # Use current_output_dir
                plot_attention_map(
                    original_image,
                    mean_head_attention_to_patches,
                    patch_size,
                    h_proc,
                    w_proc,
                    attention_output_filename
                )

                # Cleanup per image
                del original_image, inputs, outputs, last_layer_attention, attention_slice, mean_head_attention_to_patches
                if device == "cuda": torch.cuda.empty_cache()

            except Exception as e:
                import traceback
                tqdm.write(f"Error processing {image_filename} for model {model_name}: {e}")
                traceback.print_exc()
                # Explicit cleanup on error
                if 'original_image' in locals(): del original_image
                if 'inputs' in locals(): del inputs
                if 'outputs' in locals(): del outputs
                if 'last_layer_attention' in locals(): del last_layer_attention
                if device == "cuda": torch.cuda.empty_cache()
                
        # --- End Image Loop for Current Model ---
        
        # --- Cleanup After Processing Current Model --- 
        del model, config, patch_size # Remove model from memory before loading next one
        if device == "cuda": torch.cuda.empty_cache() 

    # --- End Model Configuration Loop ---
    print(f"\nComparison attention visualization finished.")
    print(f"Register model maps saved to: {OUTPUT_DIR_REG}")
    print(f"Non-register model maps saved to: {OUTPUT_DIR_NO_REG}")

if __name__ == "__main__":
    main() 