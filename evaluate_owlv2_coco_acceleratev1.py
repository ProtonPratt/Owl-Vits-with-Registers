# evaluate_owlv2_lvis.py

import torch
import os
import json
import logging
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Owlv2ForObjectDetectionWithRegisters # Use the class matching your checkpoint
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from collections import defaultdict

# --- Your Custom Dataset (Modified for Eval) ---
# Option 1: Modify LvisDetectionDataset.__getitem__
# Add 'image_id' and 'original_size' to the return dict
# Option 2: Keep LvisDetectionDataset and adapt collate_fn (Shown below)
from train_dataset import LvisDetectionDataset # Assuming this is available

# --- Configuration ---
checkpoint_dir = "./converted_owlv2_large_checkpoint/"
# checkpoint_dir = "./owlvit_registers_lvis_finetuned/checkpoint-epoch-1-best/"
# checkpoint_dir = "./owlvit_registers_lvis_finetuned/checkpoint-step-400-best/"
# checkpoint_dir = "./owlvit_registers_lvis_finetuned/checkpoint-epoch-2-best/"
val_images_dir = "./val2017/" # Directory with COCO validation images
val_annotation_file = "./lvis/lvis_v1_val_10cls_filtered1.json" # LVIS validation annotations
batch_size = 8 # Adjust based on GPU memory
num_images_to_eval = None # Set to an integer (e.g., 100) to evaluate a subset, None for all
output_file = f"owlv2_lvis_detections_{os.path.basename(os.path.normpath(checkpoint_dir))}.json" # Output JSON for COCO eval
score_threshold = 0.01 # Confidence threshold for detections
trust_remote_code_flag = False # Set to True if your checkpoint relies on custom model code from the hub

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Helper Functions ---
def load_lvis_categories_for_eval(ann_file):
    """Loads categories from LVIS annotation file, returns id_to_name and id_to_contiguous."""
    try:
        with open(ann_file, 'r') as f:
            data = json.load(f)
        if 'categories' not in data:
            logger.error(f"'categories' key not found in {ann_file}")
            return None, None, None
        # Original LVIS category ID to name
        category_map = {cat['id']: cat['name'] for cat in data['categories']}
        # List of category names in order (for text queries)
        category_names = [cat['name'] for cat in sorted(data['categories'], key=lambda x: x['id'])]
        # Mapping from original LVIS category ID to the index in category_names list
        cat_id_to_query_idx = {cat['id']: i for i, cat in enumerate(sorted(data['categories'], key=lambda x: x['id']))}
        # Mapping from query index back to original LVIS category ID
        query_idx_to_cat_id = {v: k for k, v in cat_id_to_query_idx.items()}

        logger.info(f"Loaded {len(category_map)} categories from {ann_file}")
        return category_map, category_names, query_idx_to_cat_id
    except Exception as e:
        logger.error(f"Failed to load categories from {ann_file}: {e}")
        return None, None, None

def coco_collate_fn_eval(batch, processor):
    """Collate function for evaluation: process images, pass through IDs and sizes."""
    images = [item['image'] for item in batch]
    image_ids = [item['image_id'] for item in batch] # Get image_id
    original_sizes = [item['original_size'] for item in batch] # Get original size

    # We don't need text here, it will be added per batch in the main loop
    try:
        # Only process images here
        inputs = processor(images=images, return_tensors="pt", padding="max_length")
        # Note: processor usually pads images, we need original sizes for postprocessing
    except Exception as e:
        logger.error(f"Error during processor call in collate_fn: {e}")
        return None, None, None

    return inputs, image_ids, original_sizes

# --- Modify Dataset __getitem__ slightly for eval ---
# (Alternatively, create a new dataset class inheriting from LvisDetectionDataset)
original_getitem = LvisDetectionDataset.__getitem__
def eval_getitem(self, idx):
    data = original_getitem(self, idx) # Get image and annotations dict
    img_id = self.image_ids[idx]
    img_info = self.images_info[img_id]
    pil_image = data['image']
    return {
        'image': pil_image,
        'image_id': img_id,
        'original_size': pil_image.size # (width, height)
        # We don't strictly need 'objects' for inference, but keep if needed elsewhere
    }
LvisDetectionDataset.__getitem__ = eval_getitem # Monkey-patch for simplicity

# --- Main Evaluation Logic ---
if __name__ == "__main__":
    # 1. Load Categories for Text Queries
    logger.info("Loading LVIS categories...")
    category_map, category_names, query_idx_to_cat_id = load_lvis_categories_for_eval(val_annotation_file)
    if not category_map or not category_names or not query_idx_to_cat_id:
        exit("Failed to load categories. Exiting.")
    num_classes = len(category_names)
    logger.info(f"Will use {num_classes} categories as text queries.")
    # Prepare text query list (repeated for batching later)
    text_queries = [category_names] # A list containing one list of all category names

    # 2. Load Model and Processor
    logger.info(f"Loading model and processor from: {checkpoint_dir}")
    try:
        processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=trust_remote_code_flag)
        model = Owlv2ForObjectDetectionWithRegisters.from_pretrained(checkpoint_dir, trust_remote_code=trust_remote_code_flag) # Use the class you trained with
        model.to(device)
        model.eval()
        logger.info("Model and processor loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model/processor: {e}")
        exit()

    # 3. Load Dataset and DataLoader
    logger.info("Loading validation dataset...")
    try:
        val_dataset = LvisDetectionDataset(val_annotation_file, val_images_dir)
        # Apply subset if needed
        if num_images_to_eval is not None and num_images_to_eval < len(val_dataset):
             logger.info(f"Evaluating on a subset of {num_images_to_eval} images.")
             # Note: This subset is sequential, not random. For proper eval, consider random sampling.
             # We also need to track the specific image IDs evaluated.
             subset_indices = list(range(num_images_to_eval))
             # Create a Subset dataset (better than manipulating original)
             from torch.utils.data import Subset
             original_length = len(val_dataset)
             val_dataset = Subset(val_dataset, subset_indices)
             logger.info(f"Using Subset: {len(val_dataset)} images (out of {original_length})")


        eval_collate_fn = lambda batch: coco_collate_fn_eval(batch, processor)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, # Important: No shuffle for evaluation
            num_workers=4, # Adjust as needed
            collate_fn=eval_collate_fn,
            pin_memory=True
        )
        logger.info("Validation dataloader created.")
    except Exception as e:
        logger.error(f"Failed to load dataset/dataloader: {e}")
        exit()

    # 4. Inference Loop
    logger.info("Starting inference loop...")
    coco_results = []
    processed_image_ids = set()

    progress_bar = tqdm(total=len(val_dataloader), desc="Evaluating")
    with torch.no_grad():
        for batch_data in val_dataloader:
            if batch_data is None or batch_data[0] is None:
                logger.warning("Skipping batch due to collation error.")
                progress_bar.update(1)
                continue

            inputs, image_ids, original_sizes = batch_data
            if inputs is None: # Check again after unpacking
                logger.warning("Skipping batch due to collation error (inputs are None).")
                progress_bar.update(1)
                continue

            current_batch_size = len(image_ids)
            if current_batch_size == 0:
                 progress_bar.update(1)
                 continue

            # Prepare inputs with text queries for this batch
            # The processor expects a list of texts per image, or a list of lists if all images share the same queries
            batch_text_queries = text_queries * current_batch_size # Repeat the list of all categories for each image
            try:
                 inputs_with_text = processor(text=batch_text_queries, images=inputs['pixel_values'].to(device), return_tensors="pt", padding="max_length")
                 inputs_with_text = {k: v.to(device) for k, v in inputs_with_text.items()} # Ensure all tensors are on device
            except Exception as e:
                logger.error(f"Error processing batch with text: {e}. Skipping.")
                progress_bar.update(1)
                continue

            # Model Forward Pass
            try:
                outputs = model(**inputs_with_text)
            except Exception as e:
                 logger.error(f"Error during model forward pass: {e}. Skipping batch.")
                 import traceback
                 traceback.print_exc()
                 progress_bar.update(1)
                 continue


            # Post-processing
            # target_sizes must be tensors of shape (batch_size, 2) with [height, width]
            target_sizes_tensor = torch.tensor([sz[::-1] for sz in original_sizes], device=device) # Convert (w,h) tuples to (h,w) tensors

            try:
                # The post_process_object_detection needs outputs AND target_sizes
                results_processed = processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=score_threshold,
                    target_sizes=target_sizes_tensor
                )
            except Exception as e:
                logger.error(f"Error during post-processing: {e}. Skipping batch.")
                import traceback
                traceback.print_exc()
                progress_bar.update(1)
                continue

            # Format results for COCO
            for i in range(current_batch_size):
                image_id = image_ids[i]
                processed_image_ids.add(image_id) # Track processed IDs
                scores = results_processed[i]["scores"].tolist()
                # Labels are indices corresponding to the input text_queries list
                query_indices = results_processed[i]["labels"].tolist()
                # Boxes are [xmin, ymin, xmax, ymax] absolute coordinates
                boxes_abs_xyxy = results_processed[i]["boxes"].tolist()

                for score, query_idx, box_xyxy in zip(scores, query_indices, boxes_abs_xyxy):
                    # Convert box format
                    xmin, ymin, xmax, ymax = box_xyxy
                    x = xmin
                    y = ymin
                    w = xmax - xmin
                    h = ymax - ymin
                    coco_bbox = [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]

                    # Get original LVIS category ID from query index
                    lvis_category_id = query_idx_to_cat_id.get(query_idx)
                    if lvis_category_id is None:
                        logger.warning(f"Could not map query index {query_idx} back to LVIS category ID. Skipping detection.")
                        continue

                    coco_results.append({
                        "image_id": image_id,
                        "category_id": lvis_category_id,
                        "bbox": coco_bbox,
                        "score": round(score, 3),
                    })
            progress_bar.update(1)

    progress_bar.close()
    logger.info(f"Inference finished. Processed {len(processed_image_ids)} unique images.")
    logger.info(f"Generated {len(coco_results)} detections.")

    # 5. Save Detections
    logger.info(f"Saving detections to {output_file}...")
    if not coco_results:
        logger.warning("No detections generated. Skipping saving and COCO evaluation.")
    else:
        try:
            with open(output_file, 'w') as f:
                json.dump(coco_results, f)
            logger.info("Detections saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save detections: {e}")
            exit()

        # 6. Run COCO Evaluation
        logger.info("Running COCO evaluation...")
        try:
            coco_gt = COCO(val_annotation_file)
            coco_dt = coco_gt.loadRes(output_file)

            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

            # Restrict evaluation to the images actually processed
            if num_images_to_eval is not None:
                 eval_img_ids = sorted(list(processed_image_ids))
                 logger.info(f"Evaluating metrics on {len(eval_img_ids)} specific image IDs.")
                 coco_eval.params.imgIds = eval_img_ids

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            logger.info("COCO evaluation finished.")

        except FileNotFoundError:
            logger.error(f"Annotation file not found at {val_annotation_file} for COCO evaluation.")
        except Exception as e:
            logger.error(f"Error during COCO evaluation: {e}")
            import traceback
            traceback.print_exc()

    # Restore original __getitem__ if necessary (though script exit makes it less critical)
    # LvisDetectionDataset.__getitem__ = original_getitem