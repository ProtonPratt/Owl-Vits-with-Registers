# project-hawkeye-vit
project-hawkeye-vit created by GitHub Classroom
----

### 1. **Model & Configuration**

* `transformers/Owlv2ConfigWithRegisters.py`
* `transformers/Owlv2ForObjectDetectionWithRegisters.py`
  Custom Hugging Face model/config classes enabling **register-based OWLv2 model**, a enhancement for better attention or parameter usage.
- `owlvit/transformers/src/transformers/models/owlv2` This is where our custom implementation for owl-vit with register.
``` 
.
├── configuration_owlv2.py
├── configuration_owlvit_registers.py
├── convert_owlv2_to_hf.py
├── image_processing_owlv2.py
├── __init__.py
├── modeling_owlv2.py
├── modeling_owlvit_with_registers.py
├── processing_owlv2.py
```
- We chose huggin face because most of the implementation was done and we had to make some modification to Use with Registers.

---

### 2. **Dataset & Preprocessing**

* `train_dataset.py` → `LvisDetectionDataset`
  Defines a custom PyTorch dataset class, likely used for loading **LVIS** annotations with images for training and validation.

* `category_mapping_debug.json`
  Contains class ID to name mappings for LVIS categories. Used for decoding predictions and training targets.

* `filter_lvis.py`
  Script to filter or clean the LVIS dataset — maybe to remove rare classes, balance, or format it.

---

### 3. **Loss Functions**

* `loss.py`
  Implements:

  * `HungarianMatcher`: For bipartite matching between predicted and ground-truth boxes
  * `compute_loss`: Composite loss combining classification, bounding box regression, etc.
  * `boxes_cxcywh_to_xyxy`: Box conversion utility

---

### 4. **Training Scripts**

* `train_reg_acc_loss.py`
  Main training script that:

  * Uses Accelerate for multi-GPU
  * Computes advanced loss
  * Logs metrics to WandB

---

## **Supporting Folders**

* `checkpoints/`, `checkpoint_small/`
  Stores model checkpoints (either full model or intermediate/light versions).

* `lvis/`
  Root folder likely containing LVIS annotations or class metadata.

* `train2017/`, `val2017/`, `test2017/`
  COCO-format image folders used for training/validation/testing. These are likely symlinked or referenced by the LVIS dataset.

---

## **Evaluation and Inference Scripts**

* `evaluate_owlv2_coco_accelerate.py`, `evaluate_owlv2_coco_acceleratev1.py`, `eval2_coco.py`, `eavl_coco.py`
  Evaluation logic (mean Average Precision, thresholding, etc.) using Accelerate or vanilla PyTorch

* `inference.py`
  Likely for running inference on a new set of images, producing predictions from a trained model.

* `coco_owlv2_results*.json`, `owlv2_lvis_detections_*.json`
  JSON outputs from evaluations or predictions, usually formatted for COCO-style evaluation.

---

## **Visualization and Debug Tools**

* `visualize_attention.py`, `visualize_attention_old.py`
  Visual tools to display attention maps from OWLv2.

* `visualize_detections.py`
  Probably used to overlay predicted bounding boxes on images.

* `debug_outputs/`, `attention_maps_output_*`
  Stores debug artifacts, attention visualizations, or intermediate predictions.

* `check_images.py`
  Likely validates the existence or format of image files in the dataset.

---

## **Utilities**

* `remove_missing_images.py`
  Cleans up annotation/image pairs if images are missing.

* `print_checkpoint_keys.py`
  Prints the keys in a model checkpoint — useful for debugging model loading mismatches.

* `urls.txt`
 links to models, dataset mirrors, and WANDB runs.


| Component                | Role in Training                                      |
| ------------------------ | ----------------------------------------------------- |
| `Owlv2*WithRegisters`    | Custom model architecture for detection               |
| `LvisDetectionDataset`   | Dataset loader for image/annotation pipelines         |
| `HungarianMatcher`       | Matches predictions with GT boxes for loss            |
| `compute_loss`           | Computes composite detection loss                     |
| `category_mapping*.json` | Helps interpret class indices during training/metrics |
| Accelerate & WandB       | Used for multi-GPU training & logging                 |
| `train2017`, `val2017`   | Training and validation image sets                    |
| `checkpoints/`           | Stores trained model states                           |

