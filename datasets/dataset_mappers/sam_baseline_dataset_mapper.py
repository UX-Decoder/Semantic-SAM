# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------------------
# COCO Instance Segmentation with LSJ Augmentation
# Modified from:
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py
# ------------------------------------------------------------------------------------------------

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
# from ..registration.register_sam import *
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


class SamBaselineDatasetMapper:
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
        _root = os.getenv("SAM_DATASETS", "no")
        if _root!='no':
            totoal_images = 0
            if is_train:
                self.current_tsv_id = -1
                tsv_file = f"{_root}/"
                self.tsv = {}
                print("start dataset mapper, get tsv_file from ", tsv_file)
                files = os.listdir(tsv_file)
                print('files ', files)

                start = int(os.getenv("SAM_SUBSET_START", "0"))
                end = int(os.getenv("SAM_SUBSET_END", "100"))
                if 'part' in files[0]:  # for hgx
                    files = [f for f in files if '.tsv' in f and int(f.split('.')[1].split('_')[-1])>=start and int(f.split('.')[1].split('_')[-1])<end]
                else:  # for msr
                    files = [f for f in files if '.tsv' in f and int(f.split('.')[0].split('-')[-1])>=start and int(f.split('.')[0].split('-')[-1])<end]
                self.total_tsv_num = len(files)
                for i, tsv in enumerate(files):
                    if tsv.split('.')[-1] == 'tsv':
                        self.tsv[i] = TSVFile(f"{_root}/{tsv}")
                    print("using training file ", tsv, 'files', self.tsv[i].num_rows())
                    totoal_images += self.tsv[i].num_rows()
                print('totoal_images', totoal_images)
            else:
                self.current_tsv_id = -1
                tsv_file = f"{_root}"
                self.tsv = {}
                files = os.listdir(tsv_file)
                files = [f for f in files if '.tsv' in f]
                self.total_tsv_num = len(files)
                for i, tsv in enumerate(files):
                    if tsv.split('.')[-1] == 'tsv':
                        self.tsv[i] = TSVFile(f"{_root}/{tsv}")
        else:
            if self.is_train:
                assert not self.is_train, 'can not train without SA-1B datasets, please export'
            print("Not avalible SA-1B datasets. Skip dataset mapper preparing")

        self.img_format = image_format
        self.is_train = is_train
        self.copy_flay = 0

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
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        return img

    def read_json(self, row):
        anno=json.loads(row[1])
        return anno

    def __call__(self, idx_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # self.init_copy()
        idx=idx_dict['idx']
        # if idx == 0:   # read the next tsv file now
        current_tsv_id = idx[0]
        current_idx = idx[1]
        # print('before seek ', current_tsv_id, current_idx)
        row = self.tsv[current_tsv_id].seek(current_idx)
        # print('after seed')
        dataset_dict=self.read_json(row)
        if len(dataset_dict['annotations'])==0:
            print("encounter image with empty annotations, choose the first image in the first tsv file")
            current_tsv_id = 0
            current_idx = 0
            row = self.tsv[current_tsv_id].seek(current_idx)
        dataset_dict=self.read_json(row)
            
        image = self.read_img(row)
        image = utils.convert_PIL_to_numpy(image,"RGB")
        ori_shape = image.shape[:2]
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        
        dataset_dict.update(dataset_dict['image'])
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
            
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            # if hasattr(instances, 'gt_masks'):
            #     gt_masks = instances.gt_masks
            #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            #     instances.gt_masks = gt_masks
            # import ipdb; ipdb.set_trace()
            ####
            # instances.gt_classes = torch.tensor([])
            ###
            dataset_dict["instances"] = instances
        
        ##########################
        # FIXME filter invalid data; maybe more complex filtering
        while instances.gt_classes.shape[0]==0 or mask_shape[0]!=ori_shape[0] or mask_shape[1]!=ori_shape[1]:
            print("current_tsv_id, current_idx",current_tsv_id, current_idx)
            print("no masks! instances.gt_classes.shape[0] ", instances.gt_classes.shape[0]) if instances.gt_classes.shape[0]==0 else print('wrong shape! mask_shape ', mask_shape, 'ori_shape ', ori_shape)
            current_tsv_id = random.randint(0, len(self.tsv)-1)
            current_idx = random.randint(0, self.tsv[current_tsv_id].num_rows()-1)
            row = self.tsv[current_tsv_id].seek(current_idx)
            dataset_dict=self.read_json(row)
                
            image = self.read_img(row)
            image = utils.convert_PIL_to_numpy(image,"RGB")
            ori_shape = image.shape[:2]
            # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            
            dataset_dict.update(dataset_dict['image'])
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
                
                # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
                # if hasattr(instances, 'gt_masks'):
                #     gt_masks = instances.gt_masks
                #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                #     instances.gt_masks = gt_masks
                # import ipdb; ipdb.set_trace()
                dataset_dict["instances"] = instances
                print("generate a new index to solve this problem, now instances.gt_classes.shape[0]=", instances.gt_classes.shape[0])

        return dataset_dict