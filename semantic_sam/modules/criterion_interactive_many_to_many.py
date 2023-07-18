# ------------------------------------------------------------------------
# Copyright (c) MicroSoft, Inc. and its affiliates.
# Modified from Mask DINO https://github.com/IDEA-Research/MaskDINO by Feng Li (fliay@connect.ust.hk).
# ------------------------------------------------------------------------
"""
Many to many criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from timm.loss import SoftTargetCrossEntropy
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..utils import box_ops


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
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


    return loss.mean(1).sum() / num_boxes

def iou_score_loss(inputs, targets):
    ce_loss = F.mse_loss(inputs, targets, reduction="none")
    return ce_loss

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
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
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # only match the lowest loss
    # loss = loss.view(-1, 3)
    # loss = loss.min(1)[0]
    # return loss.sum() / num_masks
    return loss

dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
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
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterionOsPartWholeM2M(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,  dn="no",dn_losses=[], panoptic_on=False, semantic_ce_loss=False, num_mask_tokens=3, iou_loss=True):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.dn = dn
        self.dn_losses = dn_losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25

        self.panoptic_on = panoptic_on
        self.semantic_ce_loss = semantic_ce_loss

        self.num_mask_tokens = num_mask_tokens
        self.iou_loss = iou_loss

    def loss_labels_ce(self, outputs, targets, indices, num_masks, layer_id=None, extra=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if layer_id > self.top_x_layers['mask']:
            return {"loss_mask_cls_0": 0}
        assert "pred_logits" in outputs
        if indices is None or len(targets) == 0:
            loss_ce = outputs['pred_logits'].sum() * 0.0
            losses = {"loss_mask_cls_0": loss_ce}
            return losses

        src_logits = outputs["pred_logits"].type(self.empty_weight.dtype)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_mask_cls_0": loss_ce}
        return losses

    def loss_labels_masked(self, outputs, targets, indices, num_boxes, log=True, layer_id=None, extra=None):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if layer_id > self.top_x_layers['mask']:
            return {"loss_mask_cls_0": 0}
        assert 'pred_logits' in outputs
        if indices is None or len(targets) == 0:
            loss_ce = outputs['pred_logits'].sum() * 0.0
            losses = {"loss_mask_cls_0": loss_ce}
            return losses

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits[idx], target_classes_onehot[idx], num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_mask_cls_0': loss_ce}

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, layer_id=None, log=True, key='gt_whole_classes', extra=None):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if self.prediction_switch is None or 'whole' not in self.prediction_switch.keys():
            if 'labels' in targets[0].keys():
                key = 'labels'
        else:
            if not self.prediction_switch['whole']:
                return {"fake_no_loss_mask_cls_0": 0.0}
            elif key not in targets[0].keys():
                # FIXME only consider batchsize=1 case
                assert len(targets) == 1
                return {"loss_mask_cls_0": 0.0 * outputs['pred_logits'].sum()}
        assert False, "not implemented"
        # src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[key][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {"loss_mask_cls_0": loss_ce}

        return losses

    def loss_labels_part(self, outputs, targets, indices, num_boxes, layer_id=None, log=True, key='gt_part_classes', extra=None):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if not self.prediction_switch['part']:
            return {"fake_no_loss_mask_part_cls_0": 0.0}
        elif key not in targets[0].keys():
            # FIXME only consider batchsize=1 case
            assert len(targets)==1
            # return {"loss_mask_whole_cls_0": 0.0*outputs['pred_logits_part'].sum()}
            return {"loss_mask_part_cls_0": 0.0*outputs['pred_logits_part'].sum()}
        # assert False, "not implemented"
        src_logits = outputs['pred_logits_part']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[key][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes_part,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {"loss_mask_part_cls_0": loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, layer_id=None, extra=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # if layer_id >= self.top_x_layers['box']:
        #     return {"loss_bbox_0": 0, "loss_giou_0": 0}
        assert 'pred_boxes' in outputs
        if indices is None or len(targets) == 0:
            loss = outputs['pred_boxes'].sum() * 0.0
            losses = {"loss_bbox_0": loss, "loss_giou_0": loss}
            return losses
        assert len(targets)==1, "not support multi batch training"
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        level_target_inds = targets[0]['level_target_inds']
        ori_boxes = targets[0]["ori_boxes"]
        target_boxes = torch.stack([ori_boxes[ind] for inds in level_target_inds for ind in inds])
        target_boxes = target_boxes[indices[0][1]]
        # target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox_0'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou_0'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, layer_id=None, extra=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # if layer_id >= self.top_x_layers['mask']:
        #     return {"loss_mask_bce_0": 0, "loss_mask_dice_0": 0}
        assert "pred_masks" in outputs
        if indices is None or len(targets) == 0:
            loss = outputs['pred_masks'].sum() * 0.0
            losses = {"loss_mask_bce_0": loss, "loss_mask_dice_0": loss}
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        level_target_inds = targets[0]['level_target_inds']
        # masks = [t["masks"] for t in targets]
        ori_masks = targets[0]["ori_masks"].to(src_masks)
        masks = [torch.stack([ori_masks[ind] for inds in level_target_inds for ind in inds])]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.type(self.empty_weight.dtype),
                lambda logits: calculate_uncertainty(logits.type(self.empty_weight.dtype)),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).type(src_masks.dtype)
            # get gt labels
            point_labels = point_sample(target_masks, point_coords, align_corners=False, ).squeeze(1)

        point_logits = point_sample(src_masks, point_coords, align_corners=False, ).squeeze(1)

        losses = {
            "loss_mask_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_mask_dice_0": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        dice_loss_all = losses["loss_mask_dice_0"]
        losses["loss_mask_dice_0"] = dice_loss_all.sum() / num_masks
        if self.iou_loss:
            src_ious = outputs["pred_ious"]
            src_ious = src_ious.flatten(1)
            src_ious = src_ious[src_idx]
            target_iou = 1 - dice_loss_all
            losses['iou_score_loss_0'] = iou_score_loss(src_ious, target_iou).sum() / num_masks

        del src_masks
        del target_masks
        return losses

    def prep_for_dn(self,mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        known_indice = mask_dict['known_indice']
        scalar,pad_size=mask_dict['scalar'],mask_dict['pad_size']
        assert pad_size % scalar==0
        single_pad=pad_size//scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes,num_tgt,single_pad,scalar

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks=None, layer_id=None, extra=None):
        loss_map = {
            'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            'labels_part': self.loss_labels_part,
            'dn_labels_part': self.loss_labels_part,
            'dn_labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels_masked,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, layer_id=layer_id, extra=extra)

    def forward(self, outputs, targets, mask_dict=None, extra=None, task='seg'):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # TODO: use different matching and loss weight when only detection
        prediction_switch = extra
        self.prediction_switch = prediction_switch
        self.dn = 'no'
        assert self.dn=='no', 'no dn loss for interactive training'

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        match_cost = ["cls", "box", "mask"]
        if task == 'det' or task == 'seg_from_teacher':
            match_cost = ["cls", "box"]
        # Retrieve the matching between the outputs of the last layer and the targets
        if self.dn is not "no" and mask_dict is not None:
            output_known_lbs_bboxes,num_tgt,single_pad,scalar = self.prep_for_dn(mask_dict)
            exc_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                exc_idx.append((output_idx, tgt_idx))
        indices = self.matcher(outputs_without_aux, targets, match_cost, extra=extra)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        level_target_inds = targets[0]['level_target_inds']
        len_level_target_inds = [len(inds) for inds in level_target_inds]

        num_masks = sum(len_level_target_inds)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs['pred_boxes'].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if task == 'det' and loss == 'masks':
                continue   # not compute mask loss for detection data only
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, layer_id=0, extra=extra))

        if self.dn != "no" and mask_dict is not None:
            l_dict={}
            for loss in self.dn_losses:
                if task == 'det' and loss == 'masks':
                    continue  # not compute mask loss for detection data only
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, exc_idx, num_masks*scalar, layer_id=0))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        elif self.dn != "no":
            l_dict = dict()
            l_dict['loss_bbox_0_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_0_dn'] = torch.as_tensor(0.).to('cuda')
            if prediction_switch['whole']:
                l_dict['loss_mask_cls_0_dn'] = torch.as_tensor(0.).to('cuda')
            if prediction_switch['part']:
                l_dict['loss_mask_part_cls_0_dn'] = torch.as_tensor(0.).to('cuda')
            if task != 'det' and 'masks' in self.dn_losses:
                l_dict['loss_mask_bce_0_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_mask_dice_0_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets, match_cost)
                for loss in self.losses:
                    if task == 'det' and loss == 'masks':
                        continue  # not compute mask loss for detection data only
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, layer_id=(i+1), extra=extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if 'interm_outputs' in outputs:
                    start = 0
                else:
                    start = 1
                if i>=start:
                    if self.dn != "no" and mask_dict is not None:
                        out_=output_known_lbs_bboxes['aux_outputs'][i]
                        l_dict = {}
                        for loss in self.dn_losses:
                            if task == 'det' and loss == 'masks':
                                continue  # not compute mask loss for detection data only
                            l_dict.update(
                                self.get_loss(loss, out_, targets, exc_idx, num_masks * scalar, layer_id=(i+1), extra=extra))
                        l_dict = {k.replace('_0', f"_{i+1}_dn"): v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    elif self.dn != "no":
                        l_dict = dict()
                        l_dict[f'loss_bbox_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_giou_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        if prediction_switch['whole']:
                            l_dict[f'loss_mask_cls_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        if prediction_switch['part']:
                            l_dict[f'loss_mask_part_cls_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        if self.dn == "seg" and task != 'det' and 'masks' in self.dn_losses:
                            l_dict[f'loss_mask_bce_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                            l_dict[f'loss_mask_dice_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets, match_cost)
            full_set = ['labels', 'masks', 'boxes']
            for loss in list(set(self.losses) and set(full_set)):
                if task == 'det' and loss == 'masks':
                    continue  # not compute mask loss for detection data only
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_masks, layer_id=-1, extra=extra)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
