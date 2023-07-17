# ------------------------------------------------------------------------
# Copyright (c) MicroSoft, Inc. and its affiliates.
# Modified from Mask DINO https://github.com/IDEA-Research/MaskDINO by Feng Li (fliay@connect.ust.hk).
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample
from ..utils.box_ops import generalized_box_iou,box_cxcywh_to_xyxy, generalized_box_iou_padded


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class M2MHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0,
                 cost_box: float = 0, cost_giou: float = 0, panoptic_on: bool = False, num_mask_tokens: int = 3):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou

        self.panoptic_on = panoptic_on

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points
        self.num_mask_tokens = num_mask_tokens

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, cost=["cls", "box", "mask"]):
        """More memory-friendly matching. Change cost to compute only certain loss in matching"""
        if 'box' in cost:
            bs, num_queries = outputs["pred_boxes"].shape[:2]
        elif 'mask' in cost:
            bs, num_queries = outputs["pred_masks"].shape[:2]
        indices = []
        device = 'cuda'
        # Iterate through batch size
        for b in range(bs):
            level_target_inds = targets[b]['level_target_inds']
            max_num_tgt_per_click = targets[b]['max_num_tgt_per_click']
            assert max_num_tgt_per_click<=self.num_mask_tokens, "targets exceed prediction number"

            len_level_target_inds = [len(inds) for inds in level_target_inds]
            cumsum_len_level_target_inds = torch.cat(
                [torch.tensor([0]), torch.tensor(len_level_target_inds).cumsum(0)]).cuda()
            tgt_ind = (torch.cat(
                [torch.arange(i * self.num_mask_tokens, (i + 1) * self.num_mask_tokens).repeat_interleave(num) for
                 i, num in enumerate(len_level_target_inds)]).long(),
                            torch.cat([torch.arange(cumsum_len_level_target_inds[i],
                                                    cumsum_len_level_target_inds[i + 1]).repeat(self.num_mask_tokens)
                                       for i in range(len(level_target_inds))]).long())

            if 'box' in cost:
                out_bbox = outputs["pred_boxes"][b]
                ori_boxes = targets[b]["ori_boxes"]
                tgt_bbox = torch.stack([ori_boxes[ind] for inds in level_target_inds for ind in inds])
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            else:
                cost_bbox = torch.tensor(0).to(device)
                cost_giou = torch.tensor(0).to(device)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if 'mask' in cost:
                out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
                # gt masks are already padded when preparing target
                ori_masks = targets[b]["ori_masks"].to(out_mask)
                tgt_mask = torch.stack([ori_masks[ind] for inds in level_target_inds for ind in inds])

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device, dtype=tgt_mask.dtype)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            else:
                cost_mask = torch.tensor(0).to(device)
                cost_dice = torch.tensor(0).to(device)

            C = (
                self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
                + self.cost_box*cost_bbox
                + self.cost_giou*cost_giou
            )
            valid_C = C[tgt_ind]
            D = torch.ones_like(C)*1000000
            D[tgt_ind] = valid_C
            D = D.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(D))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, cost=["cls", "box", "mask"], mode='default', extra={}):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if mode == 'default':
            return self.memory_efficient_forward(outputs, targets, cost)
        else:
            assert False, "Mode {} is not supported.".format(mode)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
