import json
from torch.utils.data import Dataset # Import Dataset
from PIL import Image
import os
import torch
import logging
import wandb
import json # To load category mapping if needed

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LvisDetectionDataset(Dataset):
    def __init__(self, ann_file, img_dir):
        """
        Args:
            ann_file (string): Path to the LVIS annotation file.
            img_dir (string): Directory with all the images.
        """
        self.img_dir = img_dir
        logger.info(f"Loading annotations from {ann_file}...")
        with open(ann_file, 'r') as f:
            lvis_data = json.load(f)

        self.annotations = lvis_data['annotations']
        # Create a mapping from image ID to image info (like filename)
        self.images_info = {img['id']: img for img in lvis_data['images']}
        logger.info(f"Loaded info for {len(self.images_info)} images.")

        # Group annotations by image_id for efficient lookup
        logger.info("Grouping annotations by image ID...")
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        logger.info("Annotations grouped.")

        # Only keep image IDs that have annotations and corresponding image info
        self.image_ids = []
        for img_id in self.img_to_anns.keys():
             if img_id in self.images_info:
                 # Basic check if filename exists
                 img_info = self.images_info[img_id]
                 filename = img_info.get('file_name', img_info.get('coco_url', '').split('/')[-1])
                 if filename:
                      self.image_ids.append(img_id)
             else:
                  logger.warning(f"Image ID {img_id} found in annotations but not in image info. Skipping.")

        logger.info(f"Initialized dataset with {len(self.image_ids)} images having annotations.")


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]

        # Construct image path - Check common fields for filename
        img_filename = img_info.get('file_name')
        if not img_filename and 'coco_url' in img_info:
            img_filename = img_info['coco_url'].split('/')[-1] # Extract filename from URL if needed
        if not img_filename:
            # Fallback or error if filename can't be determined
            logger.error(f"Cannot determine filename for image ID {img_id} from info: {img_info}")
            # Depending on strictness, either raise error or return None/dummy
            # Returning dummy might be problematic later. Raising is safer initially.
            raise ValueError(f"Could not determine filename for image ID {img_id}")

        img_path = os.path.join(self.img_dir, img_filename)

        try:
            # Load image as PIL Image
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image file not found at: {img_path} for image ID {img_id}")
            # Handle missing images - maybe return None and filter in collate_fn, or raise error
            raise FileNotFoundError(f"Image file not found: {img_path}")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise # Re-raise other image loading errors

        # Get grouped annotations for this image
        anns = self.img_to_anns.get(img_id, []) # Use .get for safety, though should exist from __init__

        # Extract categories and boxes
        categories = [ann['category_id'] for ann in anns]
        # IMPORTANT: LVIS uses [x,y,width,height] format. Keep it for now.
        # Normalization/conversion should happen closer to the loss calculation
        # or potentially in the collate_fn if needed for processor.
        boxes = [ann['bbox'] for ann in anns]

        # Structure the output to match what the collate_fn expects
        objects = {'category': categories, 'bbox': boxes}

        # Return the image and its grouped annotations
        return {'image': image, 'objects': objects}