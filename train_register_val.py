# train_registers.py (with LVIS and Validation Loop)

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset # Import Dataset for potential custom class
from tqdm.auto import tqdm
from transformers import AutoProcessor, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset, Image # Import Image feature type
import logging
import wandb
import json # To load category mapping if needed

# Import your custom model and config classes
# Make sure these files are in your Python path
from transformers import OwlViTConfigWithRegisters
from transformers import OwlViTForObjectDetectionWithRegisters

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.get_logger(__name__)

# --- WandB Login Check (Optional but Recommended) ---
try:
    wandb_entity = wandb.Api().default_entity
    logger.info(f"WandB logged in as default entity: {wandb_entity}")
except wandb.errors.AuthenticationError:
    logger.warning("WandB not logged in. Run 'wandb login' or set WANDB_API_KEY.")
    wandb_entity = None # Set entity to None if not logged in or use anonymous

# --- Placeholder for Loss Function ---
def compute_loss(outputs, labels, device):
    """
    Placeholder for the OWL-ViT detection loss (Bipartite Matching based).
    Args:
        outputs (OwlViTObjectDetectionOutput): Model outputs.
        labels (list[dict]): List of dictionaries, one per batch item. Each dict should
                             contain 'class_labels' (Tensor) and 'boxes' (Tensor) in the
                             format expected by the loss (e.g., cxcywh normalized).
        device: The torch device.
    Returns:
        torch.Tensor: The calculated loss value.
        dict: Dictionary containing individual loss components (optional).
    """
    # TODO: Implement or integrate the actual bipartite matching loss.
    # This is complex and requires careful handling of label formatting and matching.
    logger.warning("Using placeholder loss function. Replace with actual detection loss.")
    pred_logits = outputs.logits
    pred_boxes = outputs.pred_boxes # Shape [batch_size, num_patches, 4]

    # Minimal placeholder: Calculate sum of logits as dummy loss
    # In a real scenario, you'd pass pred_logits, pred_boxes, and formatted labels
    # to your DETR-style loss function.
    loss = pred_logits.sum() * 0.00001 # Ensure it's a scalar and requires grad
    if torch.isnan(loss) or torch.isinf(loss):
         logger.error("Placeholder loss is NaN or Inf!")
         loss = torch.tensor(0.0, device=device, requires_grad=True) # Fallback

    loss_dict = {"placeholder_loss": loss.item()}
    return loss, loss_dict
# --- End Placeholder ---


# --- LVIS Category Loading ---
def load_lvis_categories(ann_file):
    """Loads category mapping from LVIS annotation file."""
    try:
        with open(ann_file, 'r') as f:
            data = json.load(f)
        if 'categories' not in data:
            logger.warning(f"'categories' key not found in {ann_file}")
            return None
        # Create a mapping from category id to name
        category_map = {cat['id']: cat['name'] for cat in data['categories']}
        logger.info(f"Loaded {len(category_map)} categories from {ann_file}")
        return category_map
    except Exception as e:
        logger.error(f"Failed to load categories from {ann_file}: {e}")
        return None


def preprocess_lvis_batch(batch, processor, category_map, device):
    """
    Preprocesses a batch from an LVIS dataset loaded via Hugging Face `datasets`.
    Formats text queries from category names and prepares labels conceptually.
    """
    images = batch["image"]
    annotations = batch["objects"] # Assuming 'objects' contains list of annotations per image

    texts = []
    labels_for_loss = []

    if category_map is None:
        raise ValueError("LVIS category map is required for preprocessing.")

    for i in range(len(images)):
        img_annotations = annotations[i]
        if not isinstance(img_annotations, dict) or 'category' not in img_annotations or 'bbox' not in img_annotations:
             logger.warning(f"Unexpected annotation format for image {i}. Skipping label processing. Format: {type(img_annotations)}")
             texts.append(["background"]) # Add a dummy query if needed
             labels_for_loss.append({"class_labels": torch.tensor([], dtype=torch.long), "boxes": torch.tensor([], dtype=torch.float32)}) # Empty labels
             continue

        gt_categories = img_annotations['category']
        gt_boxes = img_annotations['bbox'] # Expecting List[List[float]] format

        # --- Text Query Formatting ---
        # Get unique category names present in the image
        unique_cat_ids = sorted(list(set(gt_categories)))
        image_texts = [category_map.get(cat_id, "unknown") for cat_id in unique_cat_ids]
        if not image_texts: # Handle images with no annotations if they exist
            image_texts = ["background"] # Or handle as per model requirements
        texts.append(image_texts)

        # --- Label Formatting (NEEDS HEAVY ADAPTATION FOR YOUR LOSS) ---
        # Map ground truth category IDs to the indices within the *current* image_texts list
        cat_id_to_local_idx = {cat_id: idx for idx, cat_id in enumerate(unique_cat_ids)}
        local_class_labels = torch.tensor([cat_id_to_local_idx[cat_id] for cat_id in gt_categories], dtype=torch.long)

        # Convert bounding boxes to the required format (e.g., CXCYWH normalized)
        # This assumes input bbox format needs conversion. Adapt as necessary.
        # Example: Assuming input is XYWH format relative to image size. Needs image w/h.
        # This part is highly dependent on dataset loading and loss function requirements.
        # It's often better handled in a custom Dataset or collate_fn.
        # Placeholder: Assume boxes are already in the correct format (e.g., cxcywh normalized)
        formatted_boxes = torch.tensor(gt_boxes, dtype=torch.float32) # REPLACE WITH ACTUAL FORMATTING

        labels_for_loss.append({
            "class_labels": local_class_labels,
            "boxes": formatted_boxes
        })
        # ------------------------------------

    # Process images and text queries together
    # The processor handles padding the nested list of texts correctly
    inputs = processor(text=texts, images=images, return_tensors="pt", padding="max_length")

    # Move inputs to device
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    return inputs_on_device, labels_for_loss


def evaluate(model, dataloader, processor, category_map, device, loss_fn):
    """Basic evaluation loop to calculate validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    logger.info("Running validation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            try:
                inputs_on_device, labels_for_loss = preprocess_lvis_batch(batch, processor, category_map, device)
                outputs = model(**inputs_on_device, return_dict=True)
                loss, _ = loss_fn(outputs, labels_for_loss, device) # Pass device to loss
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                logger.warning(f"Skipping validation batch due to error: {e}")
                continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train() # Set back to training mode
    return avg_loss


def train(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load LVIS Categories ---
    category_map = load_lvis_categories(args.train_ann_file)
    if category_map is None and args.do_train:
        logger.error("Failed to load training categories. Exiting.")
        return
    val_category_map = load_lvis_categories(args.val_ann_file) if args.do_eval else category_map
    if val_category_map is None and args.do_eval:
         logger.error("Failed to load validation categories. Exiting.")
         return

    # --- Initialize wandb ---
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity, # Use detected or None
        name=f"{args.run_name}-regs-{args.num_registers}-lr-{args.learning_rate}" if args.run_name else None,
        config=vars(args),
        tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        save_code=True,
        job_type="training",
    )
    logger.info(f"Initialized wandb run: {run.name} (ID: {run.id})")

    # --- Load Processor, Config, Model ---
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    logger.info(f"Loading config for {args.model_name_or_path} with {args.num_registers} registers")
    config = OwlViTConfigWithRegisters.from_pretrained(
        args.model_name_or_path,
        num_registers=args.num_registers,
    )
    logger.info(f"Loading pre-trained model {args.model_name_or_path} and adding registers")
    model = OwlViTForObjectDetectionWithRegisters.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    logger.info("Model loaded successfully.")
    wandb.watch(model, log_freq=args.logging_steps * 5) # Watch model gradients

    # --- Load Datasets ---
    # Attempt to load LVIS using datasets, mapping image files.
    # NOTE: This might require a custom loading script or dataset class for robustness.
    try:
        logger.info(f"Loading training dataset from {args.train_ann_file} with images in {args.train_image_dir}")
        # Assumes LVIS format loadable by datasets, might need 'imagefolder' or custom builder
        train_dataset = load_dataset('json', data_files=args.train_ann_file, field='annotations', split='train')
        # Map image paths - this is often the tricky part with detection datasets
        def map_image_path_train(example):
             # Assuming 'coco_url' or similar field exists linking to image, extract filename
             # THIS IS HIGHLY DATASET DEPENDENT - YOU MUST INSPECT YOUR JSON
             image_filename = os.path.basename(example.get('coco_url', 'dummy.jpg')) # Placeholder field name
             example['image'] = os.path.join(args.train_image_dir, image_filename)
             return example
        # It might be necessary to load 'images' field separately if available in JSON
        # train_dataset = train_dataset.map(map_image_path_train) # This map might be slow
        # Instead, load images on-the-fly in a custom Dataset or preprocess_batch
        # For now, we'll assume load_dataset handles image loading or pass paths
        # A better approach is often a custom PyTorch Dataset:
        # class LvisDataset(Dataset): ... implement __len__ and __getitem__ ...
        # For simplicity, we proceed assuming preprocess can handle paths or pre-loaded images

        if args.do_eval:
            logger.info(f"Loading validation dataset from {args.val_ann_file} with images in {args.val_image_dir}")
            val_dataset = load_dataset('json', data_files=args.val_ann_file, field='annotations', split='train') # Split might be 'validation'
            # val_dataset = val_dataset.map(map_image_path_val) # Map validation images similarly
            # Example using Hugging Face Image feature to load directly if paths are correct
            train_dataset = train_dataset.cast_column("image", Image()) # Tells datasets to load the image
            if args.do_eval:
                 val_dataset = val_dataset.cast_column("image", Image())

    except Exception as e:
        logger.error(f"Failed to load dataset. Check paths and format. Consider using a custom PyTorch Dataset. Error: {e}")
        wandb.finish()
        return

    # --- DataLoaders ---
    # TODO: Implement a custom collate_fn if label formatting is complex or varies between samples
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.do_eval:
        val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        val_dataloader = None

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = args.max_train_steps if args.max_train_steps > 0 else len(train_dataloader) * args.num_epochs
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # --- Training Loop ---
    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    if args.do_eval: logger.info(f"  Num validation examples = {len(val_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs} (or {args.max_train_steps} steps)")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {num_training_steps}")

    model.train()
    global_step = 0
    progress_bar = tqdm(range(num_training_steps), desc="Training Steps")

    num_epochs_to_run = args.num_epochs if args.max_train_steps <= 0 else float('inf')
    epochs_run = 0

    while epochs_run < num_epochs_to_run and (args.max_train_steps <= 0 or global_step < args.max_train_steps):
        logger.info(f"--- Starting Epoch {int(epochs_run + 1)} ---")
        model.train() # Ensure model is in training mode
        epoch_loss = 0.0
        num_batches_epoch = 0

        for step, batch in enumerate(train_dataloader):
            # Preprocess batch and move inputs to device
            try:
                # Use the appropriate category map for training
                inputs_on_device, labels_for_loss = preprocess_lvis_batch(batch, processor, category_map, device)
            except Exception as e:
                 logger.error(f"Error processing training batch {step}: {e}. Skipping batch.")
                 continue

            # Forward pass
            try:
                outputs = model(**inputs_on_device, return_dict=True)
                loss, loss_dict = compute_loss(outputs, labels_for_loss, device) # Pass device

                if torch.isnan(loss) or torch.isinf(loss):
                     logger.warning(f"NaN or Inf loss detected at step {global_step}. Skipping step.")
                     optimizer.zero_grad() # Prevent propagation of bad gradients
                     continue

                epoch_loss += loss.item()
                num_batches_epoch += 1

            except Exception as e:
                logger.error(f"Error during train forward/loss at step {global_step}: {e}. Skipping step.")
                continue

            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1

            # Logging to WandB
            if global_step % args.logging_steps == 0:
                log_data = {
                    "train/loss": loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                }
                # Add individual losses if available
                for lname, lval in loss_dict.items():
                    log_data[f"train/loss_{lname}"] = lval
                wandb.log(log_data)

                # --- Feasibility Check Print ---
                print(f"[FEASIBILITY CHECK] Step: {global_step}, Loss: {loss.item():.4f}")

            # --- Periodic Evaluation ---
            if args.do_eval and global_step % args.eval_steps == 0:
                 avg_val_loss = evaluate(model, val_dataloader, processor, val_category_map, device, compute_loss)
                 logger.info(f"Step: {global_step}, Validation Loss: {avg_val_loss:.4f}")
                 wandb.log({
                     "eval/loss": avg_val_loss,
                     "train/global_step": global_step # Log step for eval too
                 })
                 model.train() # Ensure model is back in training mode

            # Check max steps
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                logger.info(f"Reached max_train_steps: {args.max_train_steps}")
                break

        # End of Epoch Summary
        avg_epoch_loss = epoch_loss / num_batches_epoch if num_batches_epoch > 0 else 0
        logger.info(f"--- Epoch {int(epochs_run + 1)} Finished --- Avg Loss: {avg_epoch_loss:.4f} ---")
        wandb.log({
            "train/epoch_complete": epochs_run + 1,
            "train/epoch_avg_loss": avg_epoch_loss,
        })

        # Final epoch evaluation
        if args.do_eval and (args.max_train_steps <= 0 or global_step < args.max_train_steps): # Eval if epoch finished naturally
             avg_val_loss = evaluate(model, val_dataloader, processor, val_category_map, device, compute_loss)
             logger.info(f"End of Epoch {int(epochs_run + 1)}, Validation Loss: {avg_val_loss:.4f}")
             wandb.log({
                 "eval/loss": avg_val_loss,
                 "train/epoch_complete": epochs_run + 1 # Log epoch for eval too
             })

        # Check max steps again after epoch eval
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

        epochs_run += 1


    progress_bar.close()
    logger.info("Training finished.")

    # --- Saving ---
    if args.output_dir:
        logger.info(f"Saving model checkpoint to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        # Save config too
        config.save_pretrained(args.output_dir)
        logger.info("Model, processor, and config saved.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="google/owlvit-base-patch16", help="Path to pre-trained model or shortcut name")
    parser.add_argument("--num_registers", type=int, default=4, help="Number of register tokens to add.")

    # Data arguments (LVIS specific)
    parser.add_argument("--train_ann_file", type=str, default="./lvis/lvis_v1_train.json", help="Path to LVIS training annotation JSON file.")
    parser.add_argument("--train_image_dir", type=str, default="./train2017/", help="Path to the directory containing COCO training images.")
    parser.add_argument("--val_ann_file", type=str, default="./lvis/lvis_v1_val.json", help="Path to LVIS validation annotation JSON file.")
    parser.add_argument("--val_image_dir", type=str, default="./val2017/", help="Path to the directory containing COCO validation images.")
    # parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., 'coco/2017', or path to custom script).") # Replaced by specific paths
    # parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use for training (e.g., 'train', 'train+restval').") # Implicitly handled

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./owlvit_registers_lvis_finetuned", help="Output directory for checkpoints and logs.")
    parser.add_argument("--do_train", action='store_true', default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs (ignored if max_train_steps > 0).") # Increased default
    parser.add_argument("--max_train_steps", type=int, default=-1, help="Max number of training steps. Overrides num_epochs. Set > 0 for limited run.") # Default to run epochs
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU for training.") # Reduced default for smaller GPUs
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size per GPU for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--num_warmup_steps", type=int, default=500, help="Number of warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log training loss every X steps.") # Increased default
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run validation every X steps.") # Added eval frequency
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")

    # WandB arguments
    parser.add_argument("--wandb_project", type=str, default="owlvit-registers-lvis", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=wandb_entity, help="WandB entity name (username or team name, detected if logged in)") # Use detected entity
    parser.add_argument("--wandb_tags", type=str, default="lvis,finetune", help="Comma-separated list of tags for this run")
    parser.add_argument("--run_name", type=str, default=None, help="Optional custom name for the wandb run.")


    args = parser.parse_args()

    # Ensure output dir exists if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train(args)
    elif args.do_eval:
        logger.info("Running evaluation only (Note: evaluation logic currently only calculates loss).")
        # Minimal setup for eval-only (if needed later)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_category_map = load_lvis_categories(args.val_ann_file)
        if val_category_map:
            processor = AutoProcessor.from_pretrained(args.model_name_or_path)
            config = OwlViTConfigWithRegisters.from_pretrained(args.model_name_or_path, num_registers=args.num_registers)
            model = OwlViTForObjectDetectionWithRegisters.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
            model.to(device)
            # Load validation data
            val_dataset = load_dataset('json', data_files=args.val_ann_file, field='annotations', split='train').cast_column("image", Image())
            val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            avg_val_loss = evaluate(model, val_dataloader, processor, val_category_map, device, compute_loss)
            logger.info(f"Evaluation finished. Average Validation Loss: {avg_val_loss:.4f}")
        else:
            logger.error("Cannot run evaluation, failed to load validation categories.")
    else:
        logger.info("Neither --do_train nor --do_eval specified. Nothing to do.")