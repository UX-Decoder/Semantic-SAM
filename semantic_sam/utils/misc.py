# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py

# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from typing import List, Optional

import torch
import torch.distributed as dist
import torchvision
from torch import Tensor

from utils.constants import *

def get_iou(gt_masks, pred_masks, ignore_label=-1):
    rev_ignore_mask = ~(gt_masks == ignore_label)
    gt_masks = gt_masks.bool()
    n,h,w = gt_masks.shape
    intersection = ((gt_masks & pred_masks) & rev_ignore_mask).reshape(n,h*w).sum(dim=-1)
    union = ((gt_masks | pred_masks) & rev_ignore_mask).reshape(n,h*w).sum(dim=-1)
    ious = (intersection / union)
    return ious

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    elif tensor_list[0].ndim == 2:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(txt.shape) for txt in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, l = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, l), dtype=torch.bool, device=device)
        for txt, pad_txt, m in zip(tensor_list, tensor, mask):
            pad_txt[: txt.shape[0], : txt.shape[1]] = txt
            m[: txt.shape[1]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)

def _collate_and_pad_divisibility(tensor_list: list, div=32):
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.tensor([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    c,h,w = max_size
    pad_h = (div - h % div) if h % div != 0 else 0
    pad_w = (div - w % div) if w % div != 0 else 0
    max_size = (c,h+pad_h,w+pad_w)
    
    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))
    
    return padded_imgs

# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_class_names(name, background=True):
    if name is None:
        return None
    if 'refcoco' in name:
        class_names = ["noun"]
    elif 'pascal' in name:
        class_names = PASCAL_CLASSES_PART + ["background"]
    elif 'sam' in name:
        class_names = ['foreground'] + ["background"]
    elif 'coco' in name and 'pan' not in name:
        class_names = COCO_INSTANCE_CLASSES + ["background"]
    elif 'coco' in name:
        class_names = COCO_PANOPTIC_CLASSES + ["background"]
    elif 'ade20k_full' in name:
        class_names = ADE20K_847 + ["background"]
    elif 'ade' in name:
        class_names = ADE_PANOPTIC_CLASSES + ["background"]
    elif 'voc' in name:
        class_names = PASCAL_CLASSES + ["background"]
    elif 'vlp' in name:
        class_names = ["noun"]
    elif 'tsv' in name:
        class_names = ["noun"]
    elif 'phrasecut' in name:
        class_names = ["noun"]
    elif 'openimage' in name:
        class_names = ["noun"]
    elif 'imagenet' in name:
        class_names = IMAGENET_CLASSES
    elif 'context_459' in name:
        class_names = PASCAL_CONTEXT_459 + ["background"]
    elif 'context_59' in name:
        class_names = PASCAL_CONTEXT_59 + ["background"]
    elif 'context_33' in name:
        class_names = PASCAL_CONTEXT_33
    elif 'sunrgbd_37' in name:
        class_names = SUN_RGBD_37
    elif 'scannet_41' in name:
        class_names = SCAN_40
    elif 'scannet_38' in name:
        class_names = SCAN_37
    elif 'scannet_21' in name:
        class_names = SCAN_20
    elif 'object365' in name:
        class_names = OBJECT365
    elif 'lvis' in name:
        class_names = LVIS_CATEGORIES
    elif 'seginw' in name:
        class_names = SEGINW_CATEGORIES[name.replace('_train', '').replace('_val', '')] + ["background"]
    elif name == 'cityscapes_fine_sem_seg_val':
        class_names = CITYSCAPES
    elif name == 'cityscapes_fine_instance_seg_val':
        class_names = CITYSCAPES_THING + ["background"]
    elif name in ['cityscapes_fine_panoptic_val', 'cityscapes_fine_panoptic_train']:
        class_names = CITYSCAPES + ["background"]
    elif name == 'bdd10k_val_sem_seg':
        class_names = BDD_SEM
    elif name == 'bdd10k_40_panoptic_val':
        class_names = BDD_PANO
    else:
        assert False, "text dataset name {} is not defined".format(name)

    if background == False and "background" in class_names:
        class_names.pop(class_names.index("background"))

    return class_names

# TODO: add background to 
# def get_class_names(name):
#     if name is None:
#         return None
#     elif 'refcoco' in name:
#         return ["background"]
#     elif 'coco' in name:
#         return COCO_PANOPTIC_CLASSES + ["background"]
#     elif 'ade20k_full' in name:
#         return ADE20K_847 + ["background"]
#     elif 'ade' in name:
#         return ADE_PANOPTIC_CLASSES + ["background"]
#     elif 'scannet_41' in name:
#         return SCAN_40
#     elif 'scannet_21' in name:
#         return SCAN_20
#     elif 'sun' in name:
#         return SUN_RGBD_37
#     elif name == 'cityscapes_fine_sem_seg_val':
#         return CITYSCAPES + ["background"]
#     elif name == 'cityscapes_fine_instance_seg_val':
#         return CITYSCAPES_THING + ["background"]
#     elif name in ['cityscapes_fine_panoptic_val']:
#         return CITYSCAPES + ["background"]
#     elif name == 'bdd10k_val_sem_seg':
#         return BDD_SEM + ["background"]
#     elif name == 'bdd10k_40_panoptic_val':
#         return BDD_PANO + ["background"]
#     elif 'vlp' in name:
#         return ["background"]
#     else:
#         assert False, "text dataset name {} is not defined".format(name)
