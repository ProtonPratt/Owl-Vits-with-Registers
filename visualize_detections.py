import torch
import os
from PIL import Image, ImageDraw, ImageFont
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval # Import COCOeval
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from accelerate import Accelerator
import numpy as np # Needed for COCOeval results loading

# --- Configuration ---
# Path to the directory containing the CONVERTED Hugging Face checkpoint
checkpoint_dir = "./converted_owlv2_large_checkpoint"

# COCO paths
coco_images_dir = "./coco_dataset/val2017"
coco_annotation_file = "./coco_dataset/annotations/instances_val2017.json"

# Output directory for visualized images
output_dir = "./debug_outputs"

# Image IDs to visualize (Choose a few diverse images from val2017)
# Example IDs (replace with IDs you want to inspect):
image_ids_to_visualize = [289343, 3501, 389933, 535253, 82821] # Example: kitchen, street scene, person/objects, animals, indoor scene

# Confidence threshold for VISUALIZATION (adjust as needed)
visualization_threshold = 0.01 # Higher than eval threshold to reduce clutter

# Font size for labels
font_size = 16

# Colors
PRED_COLOR = "red"
GT_COLOR = "lime"

# --- End Configuration ---

def main():
    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Model and Processor ---
    accelerator.print(f"Loading model and processor from: {checkpoint_dir}")
    if not os.path.isdir(checkpoint_dir):
        accelerator.print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        return
    try:
        processor = Owlv2Processor.from_pretrained(checkpoint_dir)
        model = Owlv2ForObjectDetection.from_pretrained(checkpoint_dir)
        model = accelerator.prepare(model)
        model.eval()
        accelerator.print("Model and processor loaded successfully.")
    except Exception as e:
        accelerator.print(f"Error loading model/processor: {e}")
        return

    # --- 2. Load COCO Dataset Info ---
    accelerator.print(f"Loading COCO annotations from: {coco_annotation_file}")
    if not os.path.exists(coco_annotation_file):
        accelerator.print(f"Error: COCO annotation file not found at {coco_annotation_file}")
        return
    coco_gt = COCO(coco_annotation_file)
    category_ids = coco_gt.getCatIds()
    categories = coco_gt.loadCats(category_ids)
    coco_cat_names = {cat['id']: cat['name'] for cat in categories} # Use dict for lookup
    # Create text queries: "a photo of a [category name]"
    # Order matters for mapping output index back to category name
    ordered_cat_names = [coco_gt.loadCats(cat_id)[0]['name'] for cat_id in sorted(coco_gt.getCatIds())]
    text_queries = [[f"a photo of a {name}" for name in ordered_cat_names]]
    accelerator.print(f"Loaded {len(ordered_cat_names)} category names in order.")

    # --- 3. Load Font ---
    try:
        font = ImageFont.truetype("arial.ttf", font_size) # Or try "DejaVuSans.ttf" if arial isn't available
    except IOError:
        accelerator.print("Default font not found, using PIL default.")
        font = ImageFont.load_default()

    # List to store predictions in COCO format for the subset
    subset_coco_results = []

    # --- 4. Process and Visualize Selected Images ---
    accelerator.print(f"Processing images: {image_ids_to_visualize}")
    for img_id in tqdm(image_ids_to_visualize, disable=not accelerator.is_main_process):
        # Load image info and image
        img_info = coco_gt.loadImgs(img_id)[0]
        image_path = os.path.join(coco_images_dir, img_info['file_name'])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            accelerator.print(f"Warning: Image file not found {image_path}, skipping.")
            continue
        except Exception as e:
            accelerator.print(f"Warning: Error loading image {image_path}: {e}, skipping.")
            continue

        # Prepare inputs
        inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process (applying the visualization threshold)
        target_sizes = torch.tensor([image.size[::-1]], device=device) # (height, width)
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=visualization_threshold
        )

        # --- Draw Ground Truth Boxes --- (Draw GT first, so predictions are on top)
        draw = ImageDraw.Draw(image)
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            x, y, w, h = ann['bbox']
            cat_id = ann['category_id']
            category_name = coco_cat_names.get(cat_id, 'Unknown') # Use dict lookup
            text = f"GT: {category_name}"

            draw.rectangle([x, y, x + w, y + h], outline=GT_COLOR, width=2)
            # Optionally draw GT labels (can get cluttered)
            # try:
            #     text_bbox = draw.textbbox((x, y+h), text, font=font)
            #     text_width = text_bbox[2] - text_bbox[0]
            #     text_height = text_bbox[3] - text_bbox[1]
            # except AttributeError:
            #     text_width, text_height = draw.textsize(text, font=font)
            # draw.rectangle((x, y + h, x + text_width + 1, y + h + text_height + 1), fill=GT_COLOR)
            # draw.text((x+1, y+h+1), text, fill="black", font=font)

        # --- Process and Draw Prediction Results --- 
        i = 0 # Index for the first (and only) result
        scores = results[i]["scores"].cpu().tolist()
        labels_indices = results[i]["labels"].cpu().tolist()
        boxes = results[i]["boxes"].cpu().tolist()

        for score, label_idx, box in zip(scores, labels_indices, boxes):
            # Map label_idx back to the original COCO category ID
            category_name = ordered_cat_names[label_idx]
            # Find the COCO category ID from the name
            category_id = -1 # Default if name not found (shouldn't happen)
            for cat in categories:
                if cat['name'] == category_name:
                    category_id = cat['id']
                    break
            
            xmin, ymin, xmax, ymax = box
            coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin] # Convert to COCO format [x,y,w,h]

            # Add prediction to subset results list
            subset_coco_results.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [round(coord, 2) for coord in coco_bbox],
                "score": round(score, 3)
            })

            # --- Draw Prediction Box --- 
            draw.rectangle((xmin, ymin, xmax, ymax), outline=PRED_COLOR, width=3)
            text = f"{category_name}: {score:.2f}"
            try:
                text_bbox = draw.textbbox((xmin, ymin), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(text, font=font)
            draw.rectangle((xmin, ymin - text_height - 4, xmin + text_width + 4, ymin), fill=PRED_COLOR)
            draw.text((xmin + 2, ymin - text_height - 2), text, fill="white", font=font)

        # Save the annotated image
        output_filename = os.path.join(output_dir, f"{img_id}_detections_with_gt.jpg")
        image.save(output_filename)
        if accelerator.is_main_process:
            accelerator.print(f"Saved visualization to {output_filename}")

    accelerator.print("Visualization finished.")

    # --- 5. Run Evaluation on the Subset --- 
    if accelerator.is_main_process and subset_coco_results:
        accelerator.print("\n--- Evaluating on Subset ---")
        try:
            # Load results using the COCO API
            # Important: Use loadRes with the LIST of result dicts
            coco_dt = coco_gt.loadRes(subset_coco_results)

            # Create COCOeval object
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

            # Restrict evaluation to the visualized image IDs
            coco_eval.params.imgIds = image_ids_to_visualize

            # Run evaluation
            accelerator.print(f"Evaluating {len(subset_coco_results)} detections on {len(image_ids_to_visualize)} images...")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            accelerator.print("Subset evaluation finished.")

        except Exception as e:
            accelerator.print(f"Error during subset COCO evaluation: {e}")
    elif accelerator.is_main_process:
        accelerator.print("No results generated for the subset, skipping evaluation.")


if __name__ == "__main__":
    main() 