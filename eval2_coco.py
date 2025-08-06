# evaluate_metrics.py
import sys
import os
import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast_object_list
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import traceback
import tempfile
import pathlib

from safetensors.torch import safe_open

try:
    from transformers import Owlv2ConfigWithRegisters
    from transformers import Owlv2ForObjectDetectionWithRegisters
    print("Successfully imported custom model classes")
except ImportError as e:
    print(f"Failed to import custom model classes: {e}")
    sys.exit(1)


# --- Configuration via Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Finetuned OwlV2 model with COCO metrics.")
    parser.add_argument(
        "--finetuned_checkpoint_dir", # Renamed for clarity
        type=str,
        default="./owlvit_registers_lvis_finetuned/checkpoint-epoch-10-best",
        required=False,
        help="Directory containing the FINETUNED Accelerator state (e.g., checkpoint-epoch-10-best/).",
    )
    parser.add_argument(
        "--base_model_config_path", # Added argument
        type=str,
        default="./checkpoint_small/",
        required=False,
        help="Path to the ORIGINAL base model directory or HF identifier (e.g., ./checkpoint_small/ or google/owlv2-base-patch16) to load the initial config.json.",
    )
    parser.add_argument(
        "--image_dir", type=str, default="./val2017/" ,required=False, help="Directory containing the validation images."
    )
    parser.add_argument(
        "--ann_file", type=str, default="./lvis/lvis_v1_val_10cls_filtered.json",required=False, help="Path to the COCO/LVIS format annotation JSON file."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size PER GPU.")
    parser.add_argument(
        "--num_eval_images", type=int, default=500, help="Number of images to evaluate (-1 for all)."
    )
    parser.add_argument(
        "--output_json", type=str, default='./out_json_eval.json', help="Optional path to save detection results."
    )
    parser.add_argument("--score_threshold", type=float, default=0.05, help="Detection confidence threshold.")
    parser.add_argument(
        "--trust_remote_code", default=True ,action='store_true', help="Allow custom code execution."
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--num_registers", type=int, default=4, help="Number of registers used during training.") # Add arg
    parser.add_argument("--trained_image_size", type=int, default=960, help="Image size used during training (affects pos embeddings).") # Add arg
    args = parser.parse_args()
    return args

# --- Dataset Class (Keep as before) ---
class LvisEvalDataset(Dataset):
    def __init__(self, img_ids, coco_images_dir, img_id_to_info, accelerator):
        self.img_ids = img_ids
        self.coco_images_dir = coco_images_dir
        self.img_id_to_info = img_id_to_info # Store the mapping
        self.accelerator = accelerator

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_id_to_info.get(img_id) # Use .get() for safety

        if img_info:
            img_filename = img_info.get('file_name')
            original_size = (img_info.get('height', 0), img_info.get('width', 0))
            if img_filename is None:
                 self.accelerator.print(f"Warning: Missing 'file_name' for img_id {img_id} in img_info. Trying standard naming.")
                 img_filename = f"{img_id:012d}.jpg" # Fallback
        else:
            img_filename = f"{img_id:012d}.jpg"
            original_size = None

        img_path = os.path.join(self.coco_images_dir, img_filename)

        try:
            image = Image.open(img_path).convert("RGB")
            width_i, height_i = image.size

            if original_size is None or original_size[0] <= 0 or original_size[1] <= 0:
                 original_size = (height_i, width_i)

            if img_info and (img_info.get('height',0) != height_i or img_info.get('width',0) != width_i):
                 if img_info.get('height',0) > 0 and img_info.get('width',0) > 0 :
                     self.accelerator.print(f"Warning: Mismatch image size for {img_filename}. Ann: {img_info.get('height',0)}x{img_info.get('width',0)}, Actual: {height_i}x{width_i}. Using actual size.")
                 original_size = (height_i, width_i)

            return {"image": image, "image_id": img_id, "original_size": original_size}
        except FileNotFoundError:
            self.accelerator.print(f"ERROR: Image file not found {img_path}. Check image_dir.")
            return {"image": None, "image_id": img_id, "original_size": (0,0), "error": True, "reason": "FileNotFound"}
        except Exception as e:
            self.accelerator.print(f"Error loading/processing image {img_path}: {e}")
            return {"image": None, "image_id": img_id, "original_size": (0,0), "error": True, "reason": str(e)}


# --- Custom Collate Function (Keep as before) ---
def collate_fn(batch):
    batch = [item for item in batch if item and 'error' not in item and item['image'] is not None]
    if not batch: return None
    images = [item['image'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    return {"images": images, "image_ids": image_ids, "original_sizes": original_sizes}


def main():
    args = parse_args()
    accelerator = Accelerator()
    accelerator.print("Accelerator initialized.")

    # --- Resolve Paths ---
    finetuned_checkpoint_path = pathlib.Path(args.finetuned_checkpoint_dir).resolve()
    # Base model path can be identifier or path
    base_model_path_str = str(pathlib.Path(args.base_model_config_path).resolve() if os.path.exists(args.base_model_config_path) else args.base_model_config_path)

    accelerator.print(f"Using finetuned checkpoint path: {finetuned_checkpoint_path}")
    accelerator.print(f"Loading base config from: {base_model_path_str}")
    if not finetuned_checkpoint_path.is_dir():
        accelerator.print(f"ERROR: Finetuned checkpoint directory not found at: {finetuned_checkpoint_path}")
        sys.exit(1)

    accelerator.print(f"Evaluation parameters: {args}")

    # --- Import Custom Model Classes ---
    # # IMPORTANT: Ensure these import your ACTUAL Owlv2 register classes
    # try:
    #     # Assuming you have these classes defined based on the OwlViT ones provided
    from transformers import AutoProcessor
    #     # *** Replace with your actual Owlv2 register config/model class names ***
    #     from configuration_owlv2_registers import Owlv2ConfigWithRegisters # Assumed name
    #     from modeling_owlv2_with_registers import Owlv2ForObjectDetectionWithRegisters # Assumed name
    #     accelerator.print("Successfully imported custom Owlv2 model/config classes with registers.")
    # except ImportError as e:
    #     accelerator.print(f"Failed to import custom Owlv2 model/config classes with registers: {e}")
    #     accelerator.print("Please ensure 'configuration_owlv2_registers.py' and 'modeling_owlv2_with_registers.py' exist and are correct.")
    #     sys.exit(1)

    # --- Load Processor (from finetuned checkpoint is usually safe) ---
    try:
        accelerator.print(f"Loading processor from: {finetuned_checkpoint_path}")
        processor = AutoProcessor.from_pretrained(
            finetuned_checkpoint_path,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code
        )
        accelerator.print("Processor loaded successfully.")
    except Exception as e:
        accelerator.print(f"Error loading processor from {finetuned_checkpoint_path}: {e}")
        # Fallback to base model path for processor if needed
        try:
            accelerator.print(f"Falling back to loading processor from base path: {base_model_path_str}")
            processor = AutoProcessor.from_pretrained(
                base_model_path_str,
                local_files_only=os.path.exists(base_model_path_str), # Check if base is local
                trust_remote_code=args.trust_remote_code
            )
            accelerator.print("Processor loaded successfully from base path.")
        except Exception as e2:
            accelerator.print(f"Error loading processor from base path either: {e2}")
            traceback.print_exc()
            sys.exit(1)


    # --- Step 1: Load BASE Config using the REGISTER Config Class ---
    try:
        accelerator.print(f"Loading BASE configuration from: {base_model_path_str}")
        # Use the *Register* config class to load the *base* config file
        config = Owlv2ConfigWithRegisters.from_pretrained(
            base_model_path_str,
            local_files_only=os.path.exists(base_model_path_str), # Check if base is local path
            trust_remote_code=args.trust_remote_code,
        )
        accelerator.print("Base config loaded.")

        # --- Step 2: Modify Config ---
        accelerator.print("Modifying config for fine-tuned model...")
        # Set number of registers
        config.num_registers = args.num_registers
        if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'num_registers'):
             config.vision_config.num_registers = args.num_registers
        else:
             accelerator.print("Warning: Could not set num_registers in vision_config.")

        # Set image size used during training
        if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'image_size'):
            original_size = config.vision_config.image_size
            config.vision_config.image_size = args.trained_image_size
            if original_size != args.trained_image_size:
                 accelerator.print(f"Updated vision_config.image_size from {original_size} to {args.trained_image_size}")
        else:
            accelerator.print("Warning: Could not set image_size in vision_config.")


        # Log final config values used for initialization
        num_reg = getattr(config, 'num_registers', 'N/A')
        img_s = getattr(config, 'vision_config', {}).image_size if hasattr(config, 'vision_config') else 'N/A'
        patch_s = getattr(config, 'vision_config', {}).patch_size if hasattr(config, 'vision_config') else 'N/A'
        accelerator.print(f"Final config for init: num_registers={num_reg}, image_size={img_s}, patch_size={patch_s}")

    except Exception as e:
        accelerator.print(f"Error loading/modifying configuration: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Step 3: Initialize Model with MODIFIED Config ---
    try:
        accelerator.print("Initializing model structure using MODIFIED configuration...")
        # Use the *Register* model class and the *modified* config
        model = Owlv2ForObjectDetectionWithRegisters(config=config)

        # Optional: Log the size of the initialized position embedding
        try:
             init_pos_emb_size = model.owlv2.vision_model.embeddings.position_embedding.weight.size()
             # Expected size based on args:
             patch_grid_size = (args.trained_image_size // config.vision_config.patch_size)**2
             expected_num_pos = patch_grid_size + 1 # CLS token
             # Note: Registers don't change pos embedding size in the provided code template
             accelerator.print(f"Initialized model position embedding size: {init_pos_emb_size} (Expected num positions: {expected_num_pos})")
             if init_pos_emb_size[0] != expected_num_pos:
                 accelerator.print(f"WARNING: Initialized pos embedding size {init_pos_emb_size[0]} doesn't match expected {expected_num_pos} based on image_size and patch_size!")

        except AttributeError:
             accelerator.print("Could not access initialized position embedding size for logging.")

        accelerator.print("Model structure initialized successfully.")

    except Exception as e:
        accelerator.print(f"Error initializing model structure with modified config: {e}")
        accelerator.print("Check the __init__ methods of your custom Owlv2*WithRegisters classes and the modified config.")
        traceback.print_exc()
        sys.exit(1)

    # --- Step 4: Prepare Model with Accelerate ---
    accelerator.print("Preparing initialized model structure with Accelerate...")
    # model = accelerator.prepare(model)
    prepared_model = accelerator.prepare(model)
    accelerator.print("Model prepared.")

    # --- Step 5: Load State (Weights) from FINETUNED Checkpoint ---
    try:
        # accelerator.print(f"Loading model weights and state from finetuned checkpoint: {finetuned_checkpoint_path}")
        # accelerator.load_state(str(finetuned_checkpoint_path))
        # accelerator.print("Model weights and state loaded successfully via accelerator.load_state.")
        accelerator.print(f"Manually loading state dict from: {finetuned_checkpoint_path}")
        state_dict = {}
        model_file = finetuned_checkpoint_path / "model.safetensors"
        pytorch_bin_file = finetuned_checkpoint_path / "pytorch_model.bin"
        
        if model_file.is_file():
            accelerator.print(f"Loading weights from {model_file}")
            # Load to CPU first to avoid potential GPU memory issues during rename
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        elif pytorch_bin_file.is_file():
            accelerator.print(f"Loading weights from {pytorch_bin_file}")
            state_dict = torch.load(pytorch_bin_file, map_location="cpu")
        else:
            raise FileNotFoundError(f"Could not find weights file in {finetuned_checkpoint_path}")
        
    except RuntimeError as e:
         accelerator.print(f"FATAL: RuntimeError during load_state: {e}")
         accelerator.print("This likely means the initialized model structure (esp. position embeddings) STILL doesn't match the saved weights.")
         accelerator.print("Double-check --trained_image_size, config loading, and custom model __init__ logic.")
         sys.exit(1)
    except Exception as e:
        accelerator.print(f"Error loading state from {finetuned_checkpoint_path}: {e}")
        traceback.print_exc()
        sys.exit(1)
        
        
    accelerator.print(f"Loaded state dict with {len(state_dict)} keys.")
    
    # --- Key Renaming WORKAROUND (Checkpoint 'owlv2.' -> Model 'owlvit.') ---
    # This assumes your model code *incorrectly* uses 'owlvit.' prefix
    renamed_state_dict = {}
    renamed_count = 0
    expected_register_key_in_model = "owlvit.vision_model.registers" # Based on previous error

    for key, value in state_dict.items():
        new_key = key
        if key.startswith("owlv2."):
            # Rename to match the expected structure in the initialized model
            new_key = "owlvit." + key[len("owlv2."):]
            renamed_count += 1
        renamed_state_dict[new_key] = value

    if renamed_count > 0:
        accelerator.print(f"WORKAROUND: Renamed {renamed_count} keys from checkpoint prefix 'owlv2.' to model prefix 'owlvit.'")
        state_dict_to_load = renamed_state_dict # Use the renamed dict
    else:
         accelerator.print("No keys with 'owlv2.' prefix found in checkpoint state dict. Using original keys.")
         state_dict_to_load = state_dict # Use original if no renames happened

    # ---------------------------------------------------------------

    # Get the actual model instance (potentially unwrapped if using DDP/FSDP)
    model_to_load = accelerator.unwrap_model(prepared_model)

    # --- Load state dict with strict=False ---
    accelerator.print("Loading potentially renamed state dict into model with strict=False...")
    missing_keys, unexpected_keys = model_to_load.load_state_dict(state_dict_to_load, strict=False)

    if missing_keys:
        accelerator.print(f"Info: Missing keys during load_state_dict (expected if registers weren't saved): {missing_keys}")
        # Specifically check if the registers key is missing after potential renaming
        if expected_register_key_in_model in missing_keys:
            accelerator.print(f"-> Confirmed '{expected_register_key_in_model}' was missing. Initial random weights will be used for registers.")
        elif any(k.endswith(".registers") for k in missing_keys):
             accelerator.print(f"-> Note: Some key ending in '.registers' was missing.")
    else:
         # This case should not happen based on your checkpoint analysis
         accelerator.print("Info: No keys were missing during loading.")

    if unexpected_keys:
        accelerator.print(f"Warning: Unexpected keys found in state dict and ignored: {unexpected_keys}")

    accelerator.print("Manual state dict loading finished.")
    
    model = prepared_model

    # --- Load Annotation Data (on main process) ---
    coco_api = None
    img_ids_to_eval = []
    cat_names = []
    cat_ids = []
    img_infos = None
    if accelerator.is_main_process:
        try:
            accelerator.print(f"Loading annotations from: {args.ann_file}")
            if not os.path.exists(args.ann_file):
                 raise FileNotFoundError(f"Annotation file not found: {args.ann_file}")
            coco_api = COCO(args.ann_file)
            all_img_ids = sorted(coco_api.getImgIds())
            cat_ids = sorted(coco_api.getCatIds())
            categories = coco_api.loadCats(cat_ids)
            cat_names = [cat['name'] for cat in categories]

            if not all_img_ids:
                accelerator.print("Error: No image IDs found in the annotation file.")
                img_ids_to_eval = []
            elif args.num_eval_images > 0 and args.num_eval_images < len(all_img_ids):
                img_ids_to_eval = all_img_ids[:args.num_eval_images]
                accelerator.print(f"Limiting evaluation to the first {args.num_eval_images} images.")
            else:
                img_ids_to_eval = all_img_ids
                accelerator.print(f"Evaluating on all {len(img_ids_to_eval)} images found in {args.ann_file}.")

            accelerator.print(f"Loaded {len(cat_names)} categories.")

            if img_ids_to_eval:
                accelerator.print("Loading image info details...")
                img_infos = coco_api.loadImgs(img_ids_to_eval)
                accelerator.print(f"Loaded info for {len(img_infos)} images.")
            else:
                img_infos = []

        except Exception as e:
            accelerator.print(f"Error loading COCO annotations: {e}")
            traceback.print_exc()
            # Set lists to empty to allow broadcast, but likely exit later
            img_ids_to_eval, cat_names, cat_ids, img_infos = [], [], [], []

    # Broadcast necessary data
    objects_to_broadcast = [img_ids_to_eval, cat_names, cat_ids, img_infos]
    try:
        broadcast_object_list(objects_to_broadcast, from_process=0)
    except Exception as e:
        accelerator.print(f"Error during broadcast_object_list: {e}")
        sys.exit(1)
    img_ids, cat_names, cat_ids, img_infos = objects_to_broadcast

    # Check if critical data is missing after broadcast
    if not img_ids:
         accelerator.print("Error: No image IDs available for evaluation after broadcast. Exiting.")
         sys.exit(1)
    if not cat_names:
         accelerator.print("Error: No category names available after broadcast. Exiting.")
         sys.exit(1)


    # Create mappings (needed on all processes)
    if cat_ids:
        cat_id_map = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        reverse_cat_id_map = {i: cat_id for cat_id, i in cat_id_map.items()}
    else:
         accelerator.print("Warning: No category IDs loaded/broadcasted.")
         cat_id_map, reverse_cat_id_map = {}, {}

    # Create img_id_to_info mapping robustly
    img_id_to_info = {}
    if img_infos:
        valid_info_count = 0
        for info in img_infos:
            img_id = info.get('id')
            if img_id is not None:
                img_id_to_info[img_id] = {
                    'file_name': info.get('file_name'), # Can be None
                    'height': info.get('height', 0),
                    'width': info.get('width', 0)
                }
                valid_info_count += 1
        if accelerator.is_main_process:
            accelerator.print(f"Created img_id_to_info mapping for {valid_info_count} images.")
    elif accelerator.is_main_process:
        accelerator.print("Warning: img_infos list is None or empty. Dataset will rely on fallback naming.")


    # --- Create Dataset and DataLoader ---
    eval_dataset = LvisEvalDataset(img_ids, args.image_dir, img_id_to_info, accelerator)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True # Usually beneficial
    )

    # --- Prepare DataLoader ---
    accelerator.print("Preparing dataloader with Accelerate...")
    eval_dataloader = accelerator.prepare(eval_dataloader)
    accelerator.print("Dataloader prepared.")

    # --- Evaluation Loop ---
    all_results_list = []
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        progress_bar = tqdm(total=len(eval_dataloader), desc="Evaluating", disable=not accelerator.is_main_process)
        for step, batch in enumerate(eval_dataloader):
            if batch is None: # Skip bad batches
                if accelerator.is_main_process: progress_bar.update(1)
                accelerator.print(f"Skipping step {step} due to collation error or empty batch.")
                continue

            images = batch['images'] # List of PIL Images
            image_ids = batch['image_ids']
            original_sizes = batch['original_sizes'] # List of tuples (height, width)

            texts = [cat_names] * len(images) # Use all categories as queries for each image

            try:
                # Process inputs (processor expects PIL images)
                inputs = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding="max_length", # Important for batching
                    truncation=True      # Avoid text overflow errors
                )
                # Let accelerator handle device placement for model inputs
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()} # Usually not needed after prepare

                # Forward pass
                outputs = model(**inputs)

                # Post-process results
                # Target sizes tensor needs to be on the same device as outputs
                target_sizes = torch.tensor(original_sizes, device=outputs.logits.device)

                results_processed = processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=args.score_threshold # Apply score threshold here
                )

            except Exception as e:
                accelerator.print(f"\nError during inference/postprocessing step {step}: {e}")
                traceback.print_exc()
                # Option: Save problematic image_ids?
                if accelerator.is_main_process: progress_bar.update(1)
                continue # Skip this batch

            # Format results for COCO evaluation
            batch_results = []
            for i in range(len(image_ids)):
                image_id = image_ids[i]
                boxes = results_processed[i]["boxes"].cpu().numpy()
                scores = results_processed[i]["scores"].cpu().numpy()
                labels = results_processed[i]["labels"].cpu().numpy() # These are indices corresponding to cat_names

                for box, score, label_idx in zip(boxes, scores, labels):
                    # Convert box from [xmin, ymin, xmax, ymax] to COCO [xmin, ymin, width, height]
                    box = [float(b) for b in box]
                    coco_box = [
                        box[0],
                        box[1],
                        box[2] - box[0],
                        box[3] - box[1],
                    ]
                    # Map internal label index back to COCO category ID
                    coco_cat_id = reverse_cat_id_map.get(label_idx, -1)
                    if coco_cat_id == -1:
                        accelerator.print(f"Warning: Found label index {label_idx} not in reverse_cat_id_map for image {image_id}. This shouldn't happen if cat_names match.")
                        continue # Skip this detection

                    batch_results.append({
                        "image_id": image_id,
                        "category_id": coco_cat_id,
                        "bbox": coco_box,
                        "score": float(score),
                    })

            # Gather results from all processes
            gathered_results_nested = gather_object(batch_results)
            # # Flatten the list of lists
            # for sublist in gathered_results_nested:
            #      all_results_list.extend(sublist)
            # Combine results ONLY on the main process
            if accelerator.is_main_process:
                # Flatten the list of lists correctly
                gathered_results_flat = []
                for i, sublist in enumerate(gathered_results_nested):
                    # <<< --- START FIX --- >>>
                    # Check if the sublist is NOT empty before accessing its elements
                    if sublist:
                        # Optional: Check the type of the first element for debugging
                        if isinstance(sublist[0], dict):
                            gathered_results_flat.extend(sublist)
                        else:
                            accelerator.print(f"Warning: Unexpected format in gathered sublist from process {i} at step {step}. First item type: {type(sublist[0])}")
                    # If sublist is empty, just skip it (no results from that process for this batch)
                    # accelerator.print(f"Debug: Step {step}, added {len(gathered_results_flat)} results. Total now: {len(all_results_list)}") # Optional debug print

            all_results_list.extend(gathered_results_flat)

            if accelerator.is_main_process:
                progress_bar.update(1)

        if accelerator.is_main_process:
            progress_bar.close()

    # --- Perform COCO Evaluation (on main process) ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"\nTotal detections gathered across all processes: {len(all_results_list)}")

        # Determine output file path
        if args.output_json:
            output_file_path = args.output_json
            cleanup_output = False
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        else:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False)
            output_file_path = temp_file.name
            cleanup_output = True
            accelerator.print(f"Using temporary file for detections: {output_file_path}")


        # Save detections to the file
        accelerator.print(f"Saving {len(all_results_list)} detections to {output_file_path}...")
        try:
            with open(output_file_path, 'w') as f:
                json.dump(all_results_list, f, indent=4)
            accelerator.print(f"Saved detections successfully.")
        except Exception as e:
            accelerator.print(f"Error saving detections: {e}")
            if cleanup_output and os.path.exists(output_file_path):
                os.remove(output_file_path) # Clean up temp file on error
            sys.exit(1)

        # Run evaluation only if detections were saved and ground truth exists
        if not all_results_list:
            accelerator.print("No detections found or saved, skipping COCO evaluation.")
        elif not os.path.exists(args.ann_file):
             accelerator.print(f"Ground truth file not found at {args.ann_file}, skipping COCO evaluation.")
        else:
            try:
                # Load ground truth
                accelerator.print("Loading ground truth for evaluation...")
                coco_gt = COCO(args.ann_file)

                # Load detection results from the saved file
                accelerator.print("Loading saved detection results for evaluation...")
                coco_dt = coco_gt.loadRes(output_file_path)

                # Run COCO evaluation
                accelerator.print("Running COCO evaluation...")
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox') # 'bbox' for object detection

                # Ensure image IDs used for evaluation match those processed and have results
                eval_img_ids = sorted(list(set([res['image_id'] for res in all_results_list])))
                if not eval_img_ids:
                     accelerator.print("No valid image IDs found in detections. Cannot run evaluation.")
                else:
                    accelerator.print(f"Evaluating metrics on {len(eval_img_ids)} images with detections.")
                    # Filter GT image IDs based on evaluated set if needed, but COCOeval handles this
                    coco_eval.params.imgIds = eval_img_ids

                    coco_eval.evaluate()    # Run per image evaluation
                    coco_eval.accumulate()  # Accumulate results
                    coco_eval.summarize()   # Print the summary metrics

            except Exception as e:
                accelerator.print(f"\nError during COCO evaluation: {e}")
                traceback.print_exc()

        # Clean up temporary file if used
        if cleanup_output and os.path.exists(output_file_path):
            accelerator.print(f"Removing temporary file: {output_file_path}")
            os.remove(output_file_path)

    accelerator.print("\nEvaluation script finished.")

    accelerator.print("\nEvaluation script finished.")


if __name__ == "__main__":
    main()