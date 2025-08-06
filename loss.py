import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Dict, Tuple, Optional

# --- Dependency Helpers (Adapted/Reimplemented from Scenic/DETR) ---
# Make sure these files are in your Python path
from transformers import Owlv2ConfigWithRegisters
from transformers import Owlv2ForObjectDetectionWithRegisters
from transformers.models.owlv2.modeling_owlv2 import Owlv2ObjectDetectionOutput
from train_dataset import LvisDetectionDataset

# @torch.no_grad() decorator might be useful for helper functions if they don't need grads

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# From https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union

# From https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# Sigmoid focal loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes # Normalize by num_boxes

# --- Matcher ---
class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # Scenic uses focal loss alpha/gamma in cost calculation if enabled, DETR's default matcher doesn't.
        # We'll follow DETR's simpler approach here for the cost matrix calculation.
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["logits"].shape[:2] # Use "logits"
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1) # Use "logits"  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        # Contrary to the loss, we don't use focal loss here.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# --- Loss Computation ---

def get_loss(loss_name, outputs, targets, indices, num_boxes, **kwargs):
    loss_map = {
        'labels': loss_labels,
        'boxes': loss_boxes
    }
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss: {loss_name}")
    return loss_map[loss_name](outputs, targets, indices, num_boxes, **kwargs)

def loss_labels(outputs, targets, indices, num_boxes, log=True, focal_alpha=0.25, focal_gamma=2.0, use_focal_loss=True):
    """Classification loss (NLL or focal loss)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    assert 'logits' in outputs
    src_logits = outputs['logits']

    idx = _get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2], # Use num_classes as background label index
                                dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o

    # Add background class column to logits if it doesn't exist
    # Assuming background class is the *last* class index (num_classes)
    num_classes = src_logits.shape[-1]
    # target_classes needs indices from 0 to num_classes (inclusive)
    # src_logits has shape [B, N, num_classes] (indices 0 to num_classes-1)

    # Prepare target for CE/Focal loss: needs one-hot format [B, N, num_classes + 1]
    target_classes_onehot = F.one_hot(target_classes, num_classes + 1)

    if use_focal_loss:
         # Need logits corresponding to background class. Let's assume 0 logits for background?
         # This is a weakness - the model doesn't predict background explicitly like DETR.
         # Option 1: Add a zero logit column for background (hacky)
         zero_logits = torch.zeros((*src_logits.shape[:2], 1), device=src_logits.device)
         src_logits_with_bg = torch.cat([src_logits, zero_logits], dim=-1)

         # Option 2: Treat non-matched as background implicitly (more DETR-like)
         # This requires calculating loss only for matched pairs and using background target for non-matched.
         # Let's stick to option 1 for simplicity adapting Scenic's approach.

         loss_ce = sigmoid_focal_loss(src_logits_with_bg.transpose(1, 2),
                                      target_classes_onehot.float().transpose(1, 2),
                                      num_boxes, alpha=focal_alpha, gamma=focal_gamma) * src_logits.shape[1] # DETR scales by num_queries

    else: # Use standard cross-entropy / NLL (adapted from DETR's loss_labels)
        # The Scenic code used sigmoid_cross_entropy, implying multi-label potential.
        # Let's use binary cross entropy, summing across classes.
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot[:, :, :-1].float(), reduction='none') # Ignore last background class dim in target

        # Mask out loss for non-matched predictions implicitly via target_classes_onehot?
        # Or explicitly mask based on indices? DETR does the former via target_classes.
        # Let's simplify: calculate loss based on matched pairs only + background target?
        # This gets complicated quickly. Let's use the DETR approach for labels.

        # Reimplementing DETR's label loss logic:
        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # Compute cross-entropy. Assumes num_classes is the background index.
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reduction='none')

        # Apply weights? DETR uses eos_coef for background class weight.
        # Scenic code normalized by number of objects.
        # For simplicity, let's average over batch and queries for now. Need normalization later.
        loss_ce = loss_ce.mean() # Simplified - needs proper normalization


    losses = {'loss_ce': loss_ce}

    # TODO: Add accuracy metric calculation if log=True
    # if log:
    #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

    return losses

def loss_boxes(outputs, targets, indices, num_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    assert 'pred_boxes' in outputs
    idx = _get_src_permutation_idx(indices)
    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat([t['boxes'][J] for t, (_, J) in zip(targets, indices)], dim=0)

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes # Normalize by num_boxes

    loss_giou = 1 - torch.diag(generalized_box_iou(
        box_cxcywh_to_xyxy(src_boxes),
        box_cxcywh_to_xyxy(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes # Normalize by num_boxes
    return losses

def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


# --- Main Loss Function ---
def compute_loss(outputs: Owlv2ObjectDetectionOutput,
                 targets: List[Dict[str, torch.Tensor]],
                 matcher: nn.Module,
                 config: dict, # Dictionary holding loss config
                 device: torch.device):
    """
    Computes the final loss for OWL-ViT / Owlv2.

    Args:
        outputs (Owlv2ObjectDetectionOutput): Raw model output containing pred_logits, pred_boxes.
        targets (List[Dict]): List of dictionaries, one per batch item.
                               Each dict must contain 'labels' (Tensor[num_target_boxes])
                               and 'boxes' (Tensor[num_target_boxes, 4]). Labels are indices
                               relative to the input queries for this batch. Boxes are cxcywh normalized.
        matcher (nn.Module): Module that computes bipartite matching.
        config (dict): Dictionary containing loss hyperparameters:
                       - cost_class, cost_bbox, cost_giou (for matcher)
                       - weight_loss_ce, weight_loss_bbox, weight_loss_giou
                       - use_focal_loss (bool)
                       - focal_alpha, focal_gamma (if use_focal_loss is True)
                       - num_classes (Number of text queries + 1 for background?) - Needed for label loss
        device (torch.device): Device for tensor operations.

    Returns:
        torch.Tensor: The calculated total loss value.
        dict: Dictionary containing individual weighted loss components.
    """

    outputs_without_aux = {k: v for k, v in outputs.items() if k not in ['aux_outputs', 'enc_outputs']}

    # Retrieve the matched indices
    # Note: Matcher expects 'pred_logits' and 'pred_boxes' in outputs.
    # Targets need 'labels' and 'boxes'.
    # Ensure targets are on the correct device.
    targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
    indices = matcher(outputs_without_aux, targets_on_device)

    # Compute the average number of target boxes accross all nodes, for normalization.
    # Filter out targets with no boxes? The matcher handles empty targets.
    num_boxes = sum(len(t["labels"]) for t in targets_on_device)
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)

    # TODO: Implement distributed world size handling if using DDP/Accelerate
    # if is_dist_avail_and_initialized():
    #     torch.distributed.all_reduce(num_boxes)
    # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
    num_boxes_float = torch.clamp(num_boxes, min=1).item() # Simple version for now
    
    # Compute all the requested losses only if there are boxes to match/calculate loss for
    # This prevents errors inside loss functions with empty indices
    if num_boxes_float > 0:
        label_loss_kwargs = {
            "focal_alpha": config.get('focal_alpha', 0.25),
            "focal_gamma": config.get('focal_gamma', 2.0),
            "use_focal_loss": config.get('use_focal_loss', True)
        }
        losses = {}
        # Pass the original outputs object which contains logits/boxes
        losses.update(get_loss('labels', outputs, targets_on_device, indices, num_boxes_float, **label_loss_kwargs))
        losses.update(get_loss('boxes', outputs, targets_on_device, indices, num_boxes_float))

        # Combine losses with weights
        weight_dict = {
            'loss_ce': config.get('weight_loss_ce', 1.0),
            'loss_bbox': config.get('weight_loss_bbox', 1.0),
            'loss_giou': config.get('weight_loss_giou', 1.0)
        }
        total_loss = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)

        # Prepare loss dict for logging (with weights applied)
        loss_dict_for_logging = {k: v * weight_dict[k] for k, v in losses.items() if k in weight_dict}
        loss_dict_for_logging = {k: v.item() for k, v in loss_dict_for_logging.items()} # Convert to scalar floats

    else:
        # *** Handle the zero-box case ***
        # Create a zero loss tensor that *depends* on the model outputs
        # This ensures the graph connection for DDP, even though loss is zero.
        total_loss = (output_logits.sum() * 0.0) + (output_boxes.sum() * 0.0)
        # Log zero for individual losses
        loss_dict_for_logging = {
             'loss_ce': 0.0,
             'loss_bbox': 0.0,
             'loss_giou': 0.0
         }
        # Ensure total_loss requires grad if inputs did (should happen automatically)
        # You might need total_loss.requires_grad_(True) if outputs were detached, but shouldn't be needed here.


    return total_loss, loss_dict_for_logging

    # # Compute all the requested losses
    # losses = {}
    # # Label Loss
    # # Need to handle num_classes properly based on how Owlv2 logits work.
    # # Assuming logits are [B, N, num_queries], target labels need adapting.
    # # Let's pass use_focal_loss etc. from config
    # label_loss_kwargs = {
    #     "focal_alpha": config.get('focal_alpha', 0.25),
    #     "focal_gamma": config.get('focal_gamma', 2.0),
    #     "use_focal_loss": config.get('use_focal_loss', True) # Defaulting to focal based on scenic's compute_cost
    # }
    # losses.update(get_loss('labels', outputs, targets_on_device, indices, num_boxes, **label_loss_kwargs))

    # # Box Losses (L1 and GIoU)
    # losses.update(get_loss('boxes', outputs, targets_on_device, indices, num_boxes))


    # # Combine losses with weights
    # weight_dict = {
    #     'loss_ce': config.get('weight_loss_ce', 1.0),
    #     'loss_bbox': config.get('weight_loss_bbox', 1.0),
    #     'loss_giou': config.get('weight_loss_giou', 1.0)
    # }
    # total_loss = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)

    # # Prepare loss dict for logging (with weights applied)
    # loss_dict_for_logging = {k: v * weight_dict[k] for k, v in losses.items() if k in weight_dict}
    # loss_dict_for_logging = {k: v.item() for k, v in loss_dict_for_logging.items()} # Convert to scalar floats

    # return total_loss, loss_dict_for_logging