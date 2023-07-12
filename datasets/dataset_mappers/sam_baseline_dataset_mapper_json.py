# ------------------------------------------------------------------------
# Semantic SAM
# Copyright (c) MicroSoft, Inc. and its affiliates.
# Modified by Xueyan Zou and Jianwei Yang.
# ------------------------------------------------------------------------

import copy
import json
import logging
import os
import numpy as np
import torch
import random

from detectron2.structures import Instances, Boxes, PolygonMasks,BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask
# from ..utils.tsv
from ..utils.tsv import TSVFile, img_from_base64, generate_lineidx, FileProgressingbar
from detectron2.config import configurable


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if not is_train:
        return T.ResizeShortestEdge(
                short_edge_length=800,
                max_size=1333,
            ),
    # assert is_train, "Only support training augmentation"
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    min_scale = cfg_input['MIN_SCALE']
    max_scale = cfg_input['MAX_SCALE']

    augmentation = []

    if cfg_input['RANDOM_FLIP'] != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                vertical=cfg_input['RANDOM_FLIP'] == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


class SamBaselineDatasetMapperJSON:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentation,
        image_format,
    ):
        self.augmentation = augmentation
        logging.getLogger(__name__).info(
            "[COCO_Instance_LSJ_Augment_Dataset_Mapper] Full TransformGens used in training: {}".format(str(self.augmentation))
        )
        _root = os.getenv("SAM_DATASETS", "datasets")

        totoal_images = 0

        self.img_format = image_format
        self.is_train = is_train

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "augmentation": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
        }
        return ret
    
    def read_img(self, row):
        img = img_from_base64(row[-1])
        return img

    def read_json(selfself, row):
        anno=json.loads(row[1])
        return anno

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["img_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        ori_shape = image.shape[:2]

        # image, transforms = T.apply_transform_gens(self.augmentation, image)
        # image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        anns = json.load(open(dataset_dict["ann_name"], 'r'))['annotations']
        dataset_dict['annotations'] = anns
        
        for anno in dataset_dict['annotations']:
            anno["bbox_mode"] = BoxMode.XYWH_ABS
            anno["category_id"] = 0

        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.augmentation, image)

        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        image_shape = image.shape[:2]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        
        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            mask_shape = ori_shape
            if len(dataset_dict['annotations'])>0 and 'segmentation' in dataset_dict['annotations'][0].keys() and 'size' in dataset_dict['annotations'][0]['segmentation'].keys():
                mask_shape = dataset_dict['annotations'][0]['segmentation']['size']
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape,mask_format='bitmask')
            point_coords=[]
            for ann in annos:
                assert len(ann['point_coords'])==1
                point_coords.extend(ann['point_coords'])
            point_coords=torch.as_tensor(point_coords)
            point_coords=torch.cat([point_coords-3.,point_coords+3.],dim=1)
            point_coords=transforms.apply_box(point_coords)
            point_coords=torch.as_tensor(point_coords,device=instances.gt_boxes.device)
            instances.point_coords=point_coords
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not instances.has('gt_masks'): 
                instances.gt_masks = PolygonMasks([])  # for negative examples
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)

            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size

            dataset_dict["instances"] = instances


        return dataset_dict