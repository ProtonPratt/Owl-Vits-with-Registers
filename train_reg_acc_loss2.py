# train_registers.py (with LVIS, Validation Loop, and Accelerate)

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
from functools import partial

# Import your custom model and config classes
# Make sure these files are in your Python path
from transformers import Owlv2ConfigWithRegisters
from transformers import Owlv2ForObjectDetectionWithRegisters
from train_dataset import LvisDetectionDataset

# Import Accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger # Use Accelerate's logger
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs

from loss import HungarianMatcher, compute_loss
LOSS_IMPLEMENTATION_AVAILABLE = True

# Basic logging setup (will be enhanced by Accelerate's logger) 
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# Use Python's standard logging for startup
logging.basicConfig(level=logging.INFO)
std_logger = logging.getLogger("startup")

# --- WandB Login Check (Optional but Recommended) ---
wandb_entity = None
try:
    wandb_entity = wandb.Api().default_entity
    std_logger.info(f"WandB logged in as default entity: {wandb_entity}")
except wandb.errors.AuthenticationError:
    std_logger.warning("WandB not logged in. Run 'wandb login' or set WANDB_API_KEY.")
except Exception as e:
    std_logger.warning(f"Could not check WandB login status: {e}")

# --- End Placeholder ---


# --- LVIS Category Loading ---
def load_lvis_categories(ann_file, logger):
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

# This function is no longer needed as preprocessing happens in the collate_fn
# def preprocess_lvis_batch(batch, processor, category_map, device):
#     ...

# --- Custom Collate Function ---
def collate_fn_with_preprocessing(batch, processor, category_map, logger):
    """
    Custom collate function that preprocesses the batch using the processor.
    Handles variable-length annotations.
    Returns inputs ready for the model and labels as a list of dicts (CPU tensors).
    Accelerate's DataLoader prepare will move the model inputs to the correct device.
    """
    # 'batch' is a list of dictionaries from the dataset: [{'image': img1, 'objects': anno1}, ...]

    # Extract images and annotations from the batch list
    images = [item['image'] for item in batch]
    annotations = [item['objects'] for item in batch]

    # --- Logic moved from preprocess_lvis_batch ---
    texts = []
    labels_for_loss = [] # Will be a list of dictionaries, one per image (on CPU)

    if category_map is None:
        # This should ideally be caught earlier, but added safety
        logger.error("LVIS category map is missing in collate function.")
        raise ValueError("LVIS category map is required for preprocessing.")

    for i in range(len(images)): # Iterate through samples *before* processor call
        img_annotations = annotations[i]

        # Basic check for expected keys - adapt if your loaded data structure is different
        if not isinstance(img_annotations, dict) or 'category' not in img_annotations or 'bbox' not in img_annotations:
             logger.warning(f"Unexpected annotation format for image index {i} in batch. Skipping label processing. Format: {type(img_annotations)}")
             # Provide dummy/empty data to keep batch structure consistent
             image_texts = ["background"] # Add a dummy query
             local_class_labels = torch.tensor([], dtype=torch.long) # CPU tensor
             formatted_boxes = torch.tensor([], dtype=torch.float32) # CPU tensor
        else:
            gt_categories = img_annotations.get('category', []) # Use .get for safety
            gt_boxes = img_annotations.get('bbox', []) # Use .get for safety

            # --- Text Query Formatting ---
            unique_cat_ids = sorted(list(set(gt_categories)))
            image_texts = [category_map.get(cat_id, "unknown") for cat_id in unique_cat_ids]
            if not image_texts: # Handle images with no annotations if they exist
                image_texts = ["background"]

            # --- Label Formatting (ADAPT FOR YOUR LOSS) ---
            if unique_cat_ids and gt_categories: # Ensure gt_categories is not empty
                cat_id_to_local_idx = {cat_id: idx for idx, cat_id in enumerate(unique_cat_ids)}
                try:
                    local_class_labels = torch.tensor([cat_id_to_local_idx[cat_id] for cat_id in gt_categories], dtype=torch.long) # CPU
                    # Placeholder: Assume boxes are already in the correct format (e.g., cxcywh normalized)
                    # You might need image height/width here for normalization if boxes are absolute
                    # formatted_boxes = torch.tensor(gt_boxes, dtype=torch.float32) # CPU - REPLACE WITH ACTUAL FORMATTING
                    # --- CONVERT BOXES ---
                    converted_boxes = []
                    for box in gt_boxes: # box is [x, y, w, h] absolute
                        x, y, w, h = box
                        if img_width <= 0 or img_height <= 0: # Safety check
                            logger.warning(f"Invalid image dimensions ({img_width}, {img_height}) for image index {i}. Skipping box conversion.")
                            # Handle appropriately - maybe skip this box or the whole image's labels
                            continue # Skip this box

                        cx = (x + w / 2) / img_width
                        cy = (y + h / 2) / img_height
                        norm_w = w / img_width
                        norm_h = h / img_height
                        converted_boxes.append([cx, cy, norm_w, norm_h])

                    if converted_boxes: # Only create tensor if conversion succeeded
                        formatted_boxes = torch.tensor(converted_boxes, dtype=torch.float32) # CPU, Normalized [cx, cy, w, h]
                    else: # Handle case where no boxes could be converted
                        formatted_boxes = torch.tensor([], dtype=torch.float32).view(0, 4) # Ensure shape is (0, 4)
                except KeyError as e:
                    logger.error(f"KeyError during label creation for image index {i}: {e}. Cat IDs: {gt_categories}, Unique IDs: {unique_cat_ids}")
                    local_class_labels = torch.tensor([], dtype=torch.long) # CPU
                    formatted_boxes = torch.tensor([], dtype=torch.float32) # CPU
                except Exception as e:
                     logger.error(f"Error formatting labels for image index {i}: {e}")
                     local_class_labels = torch.tensor([], dtype=torch.long) # CPU
                     formatted_boxes = torch.tensor([], dtype=torch.float32) # CPU
                     

            else: # Handle case with no categories found for this image
                local_class_labels = torch.tensor([], dtype=torch.long) # CPU
                formatted_boxes = torch.tensor([], dtype=torch.float32) # CPU
                
            

        texts.append(image_texts)
        labels_for_loss.append({
            "class_labels": local_class_labels,
            "boxes": formatted_boxes
        })
    # --- End logic moved from preprocess_lvis_batch ---

    # Now, process the collected images and texts using the processor
    # This handles padding images and text inputs to uniform sizes within the batch
    # Returns tensors on CPU by default
    try:
        inputs = processor(text=texts, images=images, return_tensors="pt", padding="max_length")
    except Exception as e:
        logger.error(f"Error during processor call in collate_fn: {e}")
        # Handle error, maybe return dummy data or raise exception
        # For now, let's try to return something to avoid crashing the training loop immediately
        # This might need more robust handling depending on the exact error
        logger.warning("Returning None from collate_fn due to processor error.")
        return None, None # Indicate failure

    # Return the batched inputs (CPU tensors) and the list of label dicts (CPU tensors)
    return inputs, labels_for_loss


def evaluate(accelerator, model, dataloader, matcher, loss_config, logger): # Pass matcher & loss_config
    process_index = accelerator.process_index
    logger.info(f"[Rank {process_index}] Entering evaluate()...", main_process_only=False)
    """Basic evaluation loop to calculate validation loss using Accelerate and external loss."""
    if not LOSS_IMPLEMENTATION_AVAILABLE:
        logger.error("Loss implementation not available. Cannot run evaluation.", main_process_only=True)
        return 0.0 # Return dummy value

    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_losses = []

    progress_bar = tqdm(total=len(dataloader), desc="Validation",
                        disable=not accelerator.is_main_process, leave=False)

    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None or batch_data[0] is None:
                logger.warning("Skipping validation batch due to collation error.", main_process_only=False)
                continue

            try:
                inputs, labels_for_loss_list = batch_data
                # Inputs are already on the correct device

                # Move labels to device FOR THE LOSS FUNCTION
                labels_for_loss_on_device = []
                for l in labels_for_loss_list:
                     labels_for_loss_on_device.append({
                         "labels": l["class_labels"].to(accelerator.device), # Ensure key name matches loss fn expectation
                         "boxes": l["boxes"].to(accelerator.device)      # Ensure key name matches loss fn expectation
                     })
            except Exception as e:
                 logger.warning(f"Skipping validation batch due to error moving labels: {e}", main_process_only=False)
                 progress_bar.update(1)
                 continue

            try:
                # ===>>> Model Forward (NO LABELS) <<<===
                outputs = model(**inputs) # Get predictions

                # ===>>> External Loss Calculation <<<===
                loss, loss_dict = compute_loss(
                    outputs=outputs,
                    targets=labels_for_loss_on_device,
                    matcher=matcher,
                    config=loss_config,
                    device=accelerator.device
                )

                # Gather losses across all processes
                gathered_losses = accelerator.gather(loss.unsqueeze(0))
                all_losses.append(gathered_losses)

                if accelerator.is_main_process:
                    progress_bar.update(1)
                num_batches += 1 # Count effective batches processed

            except Exception as e:
                logger.warning(f"Skipping validation batch due to error during forward/loss: {e}", main_process_only=False)
                import traceback
                traceback.print_exc() # Print full traceback for debugging eval errors
                if accelerator.is_main_process:
                    progress_bar.update(1)
                continue

    # Aggregate losses on main process
    avg_loss = 0.0
    if accelerator.is_main_process:
        if all_losses:
            all_losses_tensor = torch.cat(all_losses)
            avg_loss = torch.mean(all_losses_tensor).item()
        progress_bar.close()

    model.train() # Set back to training mode
    logger.info(f"[Rank {process_index}] Exiting evaluate() successfully.", main_process_only=False)
    return avg_loss


def train(args):
    # --- Initialize Accelerator ---
    # Pass gradient_accumulation_steps if you want to use it
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    fp16 = getattr(args, 'fp16', False)
    # Pass gradient_accumulation_steps if you want to use it
    accelerator = Accelerator(log_with="wandb" if args.wandb_project else None, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]) 
    
    # Now it's safe to get the accelerate logger
    from accelerate.logging import get_logger
    logger = get_logger(__name__, log_level="INFO")
    
    logger.info(f"Accelerator state: {accelerator.state}", main_process_only=True)
    
    # --- Check if Loss Implementation is Available ---
    if not LOSS_IMPLEMENTATION_AVAILABLE:
        logger.error("Exiting because loss implementation (loss.py) is missing or invalid.", main_process_only=True)
        return

    # --- Set Seed ---
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set seed to {args.seed}", main_process_only=True)

    # --- Load LVIS Categories (only on main process, then share if needed, but collate needs it everywhere) ---
    # Let's load it on all processes for simplicity, as the collate_fn needs it.
    category_map = load_lvis_categories(args.train_ann_file, logger)
    if category_map is None and args.do_train:
        logger.error("Failed to load training categories. Exiting.", main_process_only=True)
        return
    val_category_map = load_lvis_categories(args.val_ann_file, logger) if args.do_eval else category_map
    if val_category_map is None and args.do_eval:
         logger.error("Failed to load validation categories. Exiting.", main_process_only=True)
         return

    # --- Initialize wandb (only on main process) ---
    run = None
    if accelerator.is_main_process and args.wandb_project:
        wandb_config = vars(args)
        try:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{args.run_name}-regs-{args.num_registers}-lr-{args.learning_rate}" if args.run_name else None,
                config=wandb_config,
                tags=args.wandb_tags.split(",") if args.wandb_tags else None,
                save_code=True, # Might cause issues in some setups, optional
                job_type="training",
            )
            logger.info(f"Initialized wandb run: {run.name} (ID: {run.id})", main_process_only=True)
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}. Continuing without WandB logging.", main_process_only=True)
            run = None # Ensure run is None if init fails
    # Link accelerator state to wandb AFTER init
    # accelerator.init_trackers(args.wandb_project if run else None, config=vars(args) if run else None)


    # --- Load Model and Processor (on all processes) ---
    # Add local_files_only=True for local paths
    model_path = os.path.abspath(args.model_name_or_path)
    logger.info(f"Attempting to load model components from local path: {model_path}")

    try:
        # Processor loading should generally work the same
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code # Add arg
        )
        logger.info(f"Processor loaded successfully.")

        # logger.info(f"Loading config for {model_path} with {args.num_registers} registers")
        config = Owlv2ConfigWithRegisters.from_pretrained(
            model_path,
            num_registers=args.num_registers,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code
        )
        logger.info(f"Config loaded successfully.")

        logger.info(f"Loading pre-trained model {model_path} and adding registers")
        model = Owlv2ForObjectDetectionWithRegisters.from_pretrained(
            model_path,
            config=config,
            ignore_mismatched_sizes=True, # Keep this if modifying architecture
            local_files_only=True,
            trust_remote_code=args.trust_remote_code
        )
        logger.info(f"Model loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading model/processor/config: {e}")
        # accelerator.print(f"Error loading model/processor/config: {e}") # Print error on main process
        if run and accelerator.is_main_process: wandb.finish()
        return # Exit if loading fails

    # No model.to(device) needed here

    if accelerator.is_main_process and run:
        wandb.watch(model, log_freq=args.logging_steps * 5) # Watch model gradients


    # --- FREEZE BACKBONE (Optional but Recommended) ---
    if args.freeze_backbone:
        logger.info("Freezing vision backbone parameters (owlvit.vision_model)...", main_process_only=True)
        for name, param in model.owlv2.vision_model.named_parameters(): # Adjust prefix if needed
            param.requires_grad = False
        # Optional: Log which parts are still trainable
        if accelerator.is_main_process:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Backbone frozen. Trainable params: {trainable_params} / {total_params} ({trainable_params/total_params*100:.2f}%)")

    # --- Load Datasets (on all processes) ---
    logger.info("Loading datasets using LvisDetectionDataset...")
    try:
        train_dataset = LvisDetectionDataset(args.train_ann_file, args.train_image_dir)
        logger.info(f"Training dataset loaded with {len(train_dataset)} samples.")
        if args.do_eval:
            val_dataset = LvisDetectionDataset(args.val_ann_file, args.val_image_dir)
            logger.info(f"Validation dataset loaded with {len(val_dataset)} samples.")
        else:
            val_dataset = None
    except Exception as e:
         logger.error(f"Failed to initialize LvisDetectionDataset. Check paths and JSON format. Error: {e}")
         if run and accelerator.is_main_process: wandb.finish()
         return

    # --- Create Collate Functions ---
    collate_fn_partial = partial(collate_fn_with_preprocessing,
                                 processor=processor,
                                 category_map=category_map,
                                 logger=logger)

    val_collate_fn_partial = None
    if args.do_eval and val_dataset:
        val_collate_fn_partial = partial(collate_fn_with_preprocessing,
                                       processor=processor,
                                       category_map=val_category_map,
                                       logger=logger) # Use validation map

    # --- DataLoaders ---
    logger.info("Creating DataLoaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, # Shuffle should be True for training
        num_workers=args.num_workers,
        collate_fn=collate_fn_partial,
        pin_memory=True # Often helps speed up data transfer
    )
    val_dataloader = None
    if args.do_eval and val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_collate_fn_partial,
            pin_memory=True
        )
    logger.info("DataLoaders created.")

    # --- Optimizer & Scheduler ---
    # Filter parameters that require gradients (optional but good practice)
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    #         "weight_decay": 0.0,
    #     },
    # ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # Simpler setup first

    # Calculate num_training_steps based on the dataloader AFTER prepare (for num_processes scaling)
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps) # If using grad accum
    # We calculate this later, after prepare, for accuracy in distributed setting.

    # Placeholder for scheduler num_training_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes, # Scale warmup steps
        num_training_steps=args.max_train_steps if args.max_train_steps > 0 else args.num_epochs * len(train_dataloader) # Approximate for now
    )
    
    # --- Initialize Matcher ---
    matcher = HungarianMatcher(
        cost_class=args.cost_class_coef,
        cost_bbox=args.cost_bbox_coef,
        cost_giou=args.cost_giou_coef
    )
    
    # --- Loss Configuration Dictionary ---
    loss_config = {
        'cost_class': args.cost_class_coef,
        'cost_bbox': args.cost_bbox_coef,
        'cost_giou': args.cost_giou_coef,
        'weight_loss_ce': args.weight_loss_ce,
        'weight_loss_bbox': args.weight_loss_bbox,
        'weight_loss_giou': args.weight_loss_giou,
        'use_focal_loss': args.use_focal_loss,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        # 'num_classes': num_classes_from_queries # Need to determine this dynamically per batch? Or use a fixed large number?
        # Owlv2 pred_logits shape is [B, N, num_queries]. Loss needs this.
        # Let's assume loss_labels handles this based on logit shape for now.
    }

    # --- Prepare with Accelerate ---
    logger.info("Preparing components with Accelerate...")
    # Handle val_dataloader being None
    if val_dataloader:
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
    logger.info("Accelerate preparation complete.")

    # Recalculate num_training_steps accurately after prepare
    # len(train_dataloader) now gives steps per process per epoch
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps <= 0:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        logger.info(f"Calculated max_train_steps: {args.max_train_steps} ({args.num_epochs} epochs * {num_update_steps_per_epoch} steps/epoch)", main_process_only=True)
    else:
        args.num_epochs = (args.max_train_steps + num_update_steps_per_epoch - 1) // num_update_steps_per_epoch # Calculate effective epochs
        logger.info(f"Overriding num_epochs based on max_train_steps. Effective epochs: {args.num_epochs}", main_process_only=True)


    # Now update scheduler if needed (if initial calculation was way off, though prepare might handle this)
    # It's often better to create the scheduler AFTER prepare if max_steps depends on the prepared dataloader length
    # Recreating scheduler after prepare:
    # optimizer = accelerator.prepare(optimizer) # Optimizer already prepared
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes, # Scale warmup steps by num processes
        num_training_steps=args.max_train_steps * accelerator.num_processes # Scale total steps by num processes if scheduler expects total steps across all GPUs
        # Check scheduler documentation if it expects steps per process or total steps
        # If it expects steps per process (more common): num_training_steps=args.max_train_steps
    )
    # Re-prepare the scheduler if recreated
    lr_scheduler = accelerator.prepare(lr_scheduler)


    # --- Training Loop ---
    logger.info("***** Running training *****", main_process_only=True)
    logger.info(f"  Num train examples = {len(train_dataset)}", main_process_only=True)
    if args.do_eval and val_dataset: logger.info(f"  Num validation examples = {len(val_dataset)}", main_process_only=True)
    logger.info(f"  Num Epochs = {args.num_epochs}", main_process_only=True)
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}", main_process_only=True)
    logger.info(f"  Total train batch size (w. parallel & grad accum) = {args.batch_size * accelerator.num_processes}", main_process_only=True) # Add * grad_accum_steps if used
    logger.info(f"  Total optimization steps = {args.max_train_steps}", main_process_only=True)


    model.train()
    global_step = 0
    optimizer_steps = 0
    # Progress bar only on main process
    progress_bar = tqdm(range(args.max_train_steps), desc="Training Steps", disable=not accelerator.is_main_process)

    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train() # Ensure model is in training mode at start of epoch
        epoch_loss = 0.0
        num_batches_epoch = 0 # Batches processed on this rank
        num_batches_processed_accum = 0

        for step, batch_data in enumerate(train_dataloader):
            # Stop training if max_steps is reached
            if global_step >= args.max_train_steps:
                break

            if batch_data is None or batch_data[0] is None: # Handle potential errors from collate_fn
                logger.warning(f"Skipping training step {global_step} due to collation error.")
                continue

            try:
                inputs, labels_for_loss = batch_data
                # Inputs are already on the correct device thanks to accelerator.prepare(dataloader)
                if inputs is None: # Handle collation errors
                    logger.warning(f"Skipping micro-step {global_step+1} due to collation error.")
                    global_step += 1
                    continue

                # Move labels to device FOR THE LOSS FUNCTION
                labels_for_loss_on_device = []
                for l in labels_for_loss:
                    labels_for_loss_on_device.append({
                        # Ensure keys match loss.py expectations
                        "labels": l["class_labels"].to(accelerator.device),
                        "boxes": l["boxes"].to(accelerator.device)
                    })

                # --- Model Forward (NO LABELS) ---
                outputs = model(**inputs)

                # --- External Loss Calculation ---
                loss, loss_dict = compute_loss(
                    outputs=outputs,
                    targets=labels_for_loss_on_device,
                    matcher=matcher,
                    config=loss_config,
                    device=accelerator.device
                )
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or Inf loss detected at micro-step {global_step+1}. Skipping backward.")
                    # No optimizer.zero_grad() needed inside accumulate context before step
                    global_step += 1
                    continue


                # Loss scaling and backward pass handled by accelerator
                accelerator.backward(loss)

                # Gradient clipping (optional, but recommended)
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                optimizer_steps += 1

                # Update progress bar and global step
                progress_bar.update(1)
                global_step += 1

                # Gather loss across processes for logging average batch loss
                avg_loss = accelerator.gather(loss).mean().item()
                epoch_loss += avg_loss # Accumulate average loss across all processes
                num_batches_epoch += 1

                progress_bar.set_postfix({"loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]})


                # --- Logging & Eval (Perform after optimizer step) ---
                if optimizer_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(loss).mean().item() # Gather loss for logging
                    epoch_loss += avg_loss # Accumulate *step* average loss
                    num_batches_processed_accum += 1 # Count effective batches logged

                    if accelerator.is_main_process:
                        log_data = {
                            "train/loss": avg_loss,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/optim_step": optimizer_steps,
                            "train/epoch": epoch + (step / len(train_dataloader)) # Approximate epoch progress
                        }
                        # Log individual losses from loss_dict
                        log_data.update({f"train/{k}": v for k, v in loss_dict.items()})
                        if run: wandb.log(log_data)
                    logger.info(f"[FEASIBILITY CHECK] Optim Step: {optimizer_steps}, Avg Loss: {avg_loss:.4f}", main_process_only=True)


                # --- Periodic Evaluation (Run on all processes, aggregate results) ---
                if args.do_eval and val_dataloader and optimizer_steps % args.eval_steps == 0:
                    logger.info(f"--- Starting evaluation at optim step {optimizer_steps} ---", main_process_only=True)
                    avg_val_loss = evaluate(accelerator, model, val_dataloader, matcher, loss_config, logger)
                    
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        logger.info(f"Optim Step: {optimizer_steps}, Validation Loss: {avg_val_loss:.4f}")
                        log_data = {"eval/loss": avg_val_loss, "train/optim_step": optimizer_steps}
                        if run: 
                            wandb.log(log_data)

                        # --- Saving Checkpoint ---
                        # ... (keep saving logic, trigger based on optimizer_steps) ...
                        # Use accelerator.save_state for saving
                        if args.output_dir and avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            logger.info(f"New best validation loss: {best_val_loss:.4f} at step {optimizer_steps}. Saving model...", main_process_only=True)
                            # accelerator.wait_for_everyone()
                            save_path = os.path.join(args.output_dir, f"checkpoint-step-{optimizer_steps}-best")
                            logger.info(f"Saving model state to {save_path}")
                            accelerator.save_state(save_path)
                            # Save processor separately
                            if accelerator.is_main_process:
                                processor.save_pretrained(save_path)
                            logger.info(f"Model state and processor saved to {save_path}", main_process_only=True)

                    accelerator.wait_for_everyone()
                    
                    # Ensure model is back in training mode after eval
                    model.train()


            except Exception as e:
                logger.error(f"Error during training step {global_step}: {e}. Skipping batch.")
                # Consider adding more specific error handling or traceback logging
                import traceback
                traceback.print_exc()
                optimizer.zero_grad() # Ensure grads are cleared even if step fails
                # Optionally update progress bar even on error?
                # progress_bar.update(1)
                # global_step += 1 # Decide if a failed step counts
                continue # Skip to next batch


            # Check max steps again inside inner loop
            if global_step >= args.max_train_steps:
                break
            
        # End of Epoch Summary (on main process)
        avg_epoch_loss = epoch_loss / num_batches_epoch if num_batches_epoch > 0 else 0
        if accelerator.is_main_process:
            logger.info(f"--- Epoch {epoch + 1} Finished --- Avg Train Loss: {avg_epoch_loss:.4f} ---")
            if run:
                wandb.log({
                    "train/epoch_complete": epoch + 1,
                    "train/epoch_avg_loss": avg_epoch_loss,
                })

        # Final epoch evaluation? (Already handled by periodic eval, but could add one here if needed)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # --- Saving Checkpoint ---
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}-best")
            # logger.info(f"Saving model state to {save_path}")
            accelerator.save_state(save_path)
            # Save processor separately
            processor.save_pretrained(save_path)
            logger.info(f"Model state and processor saved - {save_path}", main_process_only=True)
        accelerator.wait_for_everyone()

        # Exit loop if max steps reached after epoch finishes
        if global_step >= args.max_train_steps:
            logger.info(f"Reached max_train_steps: {args.max_train_steps} during epoch {epoch+1}", main_process_only=True)
            break


    progress_bar.close()
    logger.info("Training finished.", main_process_only=True)

    # --- Final Saving (only on main process) ---
    accelerator.wait_for_everyone()
    if args.output_dir and accelerator.is_main_process:
        logger.info(f"Saving final model checkpoint to {args.output_dir}")
        # accelerator.wait_for_everyone() # Ensure all processes are done
        unwrapped_model = accelerator.unwrap_model(model)
        try:
            final_save_path = os.path.join(args.output_dir, f"final_checkpoint")
            os.makedirs(final_save_path, exist_ok=True)
            # unwrapped_model.save_pretrained(final_save_path)
            # processor.save_pretrained(final_save_path)
            # Save config? (already included in model save)
            accelerator.save_state(final_save_path) # Optional: save optimizer/scheduler states etc.
            logger.info(f"Final model, processor saved to {final_save_path}")
        except Exception as save_err:
            logger.error(f"Error saving final checkpoint: {save_err}")

    accelerator.wait_for_everyone()
    
    if run and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="google/owlvit-base-patch16", help="Path to pre-trained model or shortcut name")
    parser.add_argument("--num_registers", type=int, default=4, help="Number of register tokens to add.")
    parser.add_argument("--trust_remote_code", action='store_true', help="Allow executing custom code from model hub")


    # Data arguments (LVIS specific)
    parser.add_argument("--train_ann_file", type=str, default="./lvis/lvis_v1_train_10cls.json", help="Path to LVIS training annotation JSON file.")
    parser.add_argument("--train_image_dir", type=str, default="./train2017/", help="Path to the directory containing COCO training images.")
    parser.add_argument("--val_ann_file", type=str, default="./lvis/lvis_v1_val_10cls_filtered.json", help="Path to LVIS validation annotation JSON file.")
    parser.add_argument("--val_image_dir", type=str, default="./val2017/", help="Path to the directory containing COCO validation images.")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./owlvit_registers_lvis_finetuned", help="Output directory for checkpoints and logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--do_train", action='store_true', default=True, help="Whether to run training.") # Keep default True
    parser.add_argument("--do_eval", action='store_true', default=True, help="Whether to run eval on the dev set.") # Keep default True
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs (used if max_train_steps <= 0).")
    parser.add_argument("--max_train_steps", type=int, default=-1, help="Max number of training steps. Overrides num_epochs. Set > 0 for limited run.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size PER GPU for training.") # Clarified per GPU
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size PER GPU for evaluation.") # Clarified per GPU
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--num_warmup_steps", type=int, default=500, help="Number of warmup steps (per process).")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log training loss every X steps.")
    parser.add_argument("--eval_steps", type=int, default=400, help="Run validation every X steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping. Set to 0 or negative to disable.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--freeze_backbone", action='store_true', help="Freeze the vision backbone during training.")

    # WandB arguments
    parser.add_argument("--wandb_project", type=str, default="owlvit-registers-lvis", help="WandB project name. Set to empty or None to disable.")
    parser.add_argument("--wandb_entity", type=str, default=wandb_entity, help="WandB entity name (username or team name, detected if logged in)")
    parser.add_argument("--wandb_tags", type=str, default="lvis,finetune,accelerate", help="Comma-separated list of tags for this run")
    parser.add_argument("--run_name", type=str, default=None, help="Optional custom name for the wandb run.")
    
    # --- Add arguments for Loss & Matcher ---
    parser.add_argument("--cost_class_coef", type=float, default=1.0, help="Matcher cost coefficient for class")
    parser.add_argument("--cost_bbox_coef", type=float, default=5.0, help="Matcher cost coefficient for L1 bbox")
    parser.add_argument("--cost_giou_coef", type=float, default=2.0, help="Matcher cost coefficient for GIoU bbox")
    parser.add_argument("--weight_loss_ce", type=float, default=1.0, help="Loss weight for classification")
    parser.add_argument("--weight_loss_bbox", type=float, default=5.0, help="Loss weight for L1 bbox")
    parser.add_argument("--weight_loss_giou", type=float, default=2.0, help="Loss weight for GIoU bbox")
    parser.add_argument("--use_focal_loss", action='store_true', default=True, help="Use Sigmoid Focal Loss for classification") # Default True based on Scenic
    parser.add_argument("--no_focal_loss", action='store_false', dest='use_focal_loss', help="Use standard BCE Loss for classification")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Alpha for Focal Loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for Focal Loss")


    args = parser.parse_args()

    # Disable WandB if project name is empty or not provided
    if not args.wandb_project:
        print("WandB project name not provided, disabling WandB logging.")
        args.wandb_project = None # Ensure it's None if empty string
        os.environ["WANDB_DISABLED"] = "true" # Also set env variable


    # Ensure output dir exists (only need on main process, but harmless elsewhere)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train(args)
    elif args.do_eval:
        # Evaluation-only mode with Accelerate is a bit more complex to set up cleanly here.
        # The current structure focuses on training. For eval-only, you'd still need
        # to initialize Accelerator, prepare model and dataloader, and run the evaluate function.
        print("Running evaluation only is not fully implemented in this script structure. Please run with --do_train (and optionally --max_train_steps=0 or 1 for minimal training).")
        # If you *really* need eval-only, adapt the beginning of the `train` function:
        # 1. Initialize Accelerator
        # 2. Load categories, processor, config, model
        # 3. Load val_dataset, create val_dataloader
        # 4. Prepare model and val_dataloader
        # 5. Call the evaluate function
        # Example sketch (untested):
        # accelerator = Accelerator()
        # val_category_map = load_lvis_categories(args.val_ann_file)
        # if val_category_map:
        #     processor = AutoProcessor.from_pretrained(...)
        #     config = Owlv2ConfigWithRegisters.from_pretrained(...)
        #     model = Owlv2ForObjectDetectionWithRegisters.from_pretrained(...)
        #     val_dataset = LvisDetectionDataset(...)
        #     val_collate_fn = partial(...)
        #     val_dataloader = DataLoader(val_dataset, ..., collate_fn=val_collate_fn)
        #     model, val_dataloader = accelerator.prepare(model, val_dataloader)
        #     avg_val_loss = evaluate(accelerator, model, val_dataloader, processor, val_category_map, accelerator.device, compute_loss)
        #     if accelerator.is_main_process: logger.info(f"Evaluation finished. Avg Loss: {avg_val_loss:.4f}")
        # else: logger.error("Cannot run eval, failed to load categories", main_process_only=True)

    else:
        print("Neither --do_train nor --do_eval specified. Nothing to do.")