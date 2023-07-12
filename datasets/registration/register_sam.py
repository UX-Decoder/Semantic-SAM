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
# Copyright (c) Facebook, Inc. and its affiliates.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_ade20k_instance.py
# ------------------------------------------------------------------------------------------------

import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
import detectron2.utils.comm as comm
import torch.distributed as dist

import os.path as op

SAM_CATEGORIES = [{'id': 1, 'name': 'stuff'}]#, {'id': 8, 'name': 'windowpane'}, {'id': 10, 'name': 'cabinet'}, {'id': 12, 'name': 'person'}, {'id': 14, 'name': 'door'}, {'id': 15, 'name': 'table'}, {'id': 18, 'name': 'curtain'}, {'id': 19, 'name': 'chair'}, {'id': 20, 'name': 'car'}, {'id': 22, 'name': 'painting'}, {'id': 23, 'name': 'sofa'}, {'id': 24, 'name': 'shelf'}, {'id': 27, 'name': 'mirror'}, {'id': 30, 'name': 'armchair'}, {'id': 31, 'name': 'seat'}, {'id': 32, 'name': 'fence'}, {'id': 33, 'name': 'desk'}, {'id': 35, 'name': 'wardrobe'}, {'id': 36, 'name': 'lamp'}, {'id': 37, 'name': 'bathtub'}, {'id': 38, 'name': 'railing'}, {'id': 39, 'name': 'cushion'}, {'id': 41, 'name': 'box'}, {'id': 42, 'name': 'column'}, {'id': 43, 'name': 'signboard'}, {'id': 44, 'name': 'chest of drawers'}, {'id': 45, 'name': 'counter'}, {'id': 47, 'name': 'sink'}, {'id': 49, 'name': 'fireplace'}, {'id': 50, 'name': 'refrigerator'}, {'id': 53, 'name': 'stairs'}, {'id': 55, 'name': 'case'}, {'id': 56, 'name': 'pool table'}, {'id': 57, 'name': 'pillow'}, {'id': 58, 'name': 'screen door'}, {'id': 62, 'name': 'bookcase'}, {'id': 64, 'name': 'coffee table'}, {'id': 65, 'name': 'toilet'}, {'id': 66, 'name': 'flower'}, {'id': 67, 'name': 'book'}, {'id': 69, 'name': 'bench'}, {'id': 70, 'name': 'countertop'}, {'id': 71, 'name': 'stove'}, {'id': 72, 'name': 'palm'}, {'id': 73, 'name': 'kitchen island'}, {'id': 74, 'name': 'computer'}, {'id': 75, 'name': 'swivel chair'}, {'id': 76, 'name': 'boat'}, {'id': 78, 'name': 'arcade machine'}, {'id': 80, 'name': 'bus'}, {'id': 81, 'name': 'towel'}, {'id': 82, 'name': 'light'}, {'id': 83, 'name': 'truck'}, {'id': 85, 'name': 'chandelier'}, {'id': 86, 'name': 'awning'}, {'id': 87, 'name': 'streetlight'}, {'id': 88, 'name': 'booth'}, {'id': 89, 'name': 'television receiver'}, {'id': 90, 'name': 'airplane'}, {'id': 92, 'name': 'apparel'}, {'id': 93, 'name': 'pole'}, {'id': 95, 'name': 'bannister'}, {'id': 97, 'name': 'ottoman'}, {'id': 98, 'name': 'bottle'}, {'id': 102, 'name': 'van'}, {'id': 103, 'name': 'ship'}, {'id': 104, 'name': 'fountain'}, {'id': 107, 'name': 'washer'}, {'id': 108, 'name': 'plaything'}, {'id': 110, 'name': 'stool'}, {'id': 111, 'name': 'barrel'}, {'id': 112, 'name': 'basket'}, {'id': 115, 'name': 'bag'}, {'id': 116, 'name': 'minibike'}, {'id': 118, 'name': 'oven'}, {'id': 119, 'name': 'ball'}, {'id': 120, 'name': 'food'}, {'id': 121, 'name': 'step'}, {'id': 123, 'name': 'trade name'}, {'id': 124, 'name': 'microwave'}, {'id': 125, 'name': 'pot'}, {'id': 126, 'name': 'animal'}, {'id': 127, 'name': 'bicycle'}, {'id': 129, 'name': 'dishwasher'}, {'id': 130, 'name': 'screen'}, {'id': 132, 'name': 'sculpture'}, {'id': 133, 'name': 'hood'}, {'id': 134, 'name': 'sconce'}, {'id': 135, 'name': 'vase'}, {'id': 136, 'name': 'traffic light'}, {'id': 137, 'name': 'tray'}, {'id': 138, 'name': 'ashcan'}, {'id': 139, 'name': 'fan'}, {'id': 142, 'name': 'plate'}, {'id': 143, 'name': 'monitor'}, {'id': 144, 'name': 'bulletin board'}, {'id': 146, 'name': 'radiator'}, {'id': 147, 'name': 'glass'}, {'id': 148, 'name': 'clock'}, {'id': 149, 'name': 'flag'}]

_PREDEFINED_SPLITS = {
    # point annotations without masks
    "sam_train": (
        "",
        # "ADEChallengeData2016/ade20k_instance_train.json",
    ),
    "sam_val": (
        "",
        # "ADEChallengeData2016/ade20k_instance_train.json",
    ),
}


def _get_sam_instances_meta():
    thing_ids = [k["id"] for k in SAM_CATEGORIES]
    assert len(thing_ids) == 1, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SAM_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def load_sam_index(tsv_file, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.
    """
    dataset_dicts = []
    tsv_id = 0
    files = os.listdir(tsv_file)
    start = int(os.getenv("SAM_SUBSET_START", "90"))
    end = int(os.getenv("SAM_SUBSET_END", "100"))
    if len(files)>0 and 'part' in files[0]:  # for hgx
                files = [f for f in files if '.tsv' in f and int(f.split('.')[1].split('_')[-1])>=start and int(f.split('.')[1].split('_')[-1])<end]
    else:  # for msr
        files = [f for f in files if '.tsv' in f and int(f.split('.')[0].split('-')[-1])>=start and int(f.split('.')[0].split('-')[-1])<end]
    # _root_local = os.getenv("SAM_LOCAL", "no")
    # azcopy = _root_local!='no'
        
    for tsv in files:
        if op.splitext(tsv)[1] == '.tsv':
            print('register tsv to create index', "tsv_id", tsv_id, tsv)
            lineidx = os.path.join(tsv_file, op.splitext(tsv)[0] + '.lineidx')
            line_name = op.splitext(tsv)[0] + '.lineidx'
            
            with open(lineidx, 'r') as fp:
                lines = fp.readlines()
                _lineidx = [int(i.strip().split()[0]) for i in lines]

            dataset_dict =[{'idx': (tsv_id, i)} for i in range(len(_lineidx))]
            dataset_dicts = dataset_dicts + dataset_dict
            tsv_id += 1
    return dataset_dicts

# def azcopy_sam_tsv(tsv_file):
#     """
#     copy tsv to local
#     """
#     tsv_id = 0
#     files = os.listdir(tsv_file)
#     start = int(os.getenv("SAM_SUBSET_START", "90"))
#     end = int(os.getenv("SAM_SUBSET_END", "100"))
#     files = [f for f in files if '.tsv' in f and int(f.split('.')[0].split('-')[-1])>=start and int(f.split('.')[0].split('-')[-1])<end]
#     _root_local = os.getenv("SAM_LOCAL", "no")
#     azcopy = _root_local!='no'
#     if azcopy:
#         if azcopy and comm.is_main_process():
#             # print("dist.is_initialized() ", dist.is_initialized())
#             # print("dist.get_rank() ", dist.get_rank())
#             print("dowload azcopy")
#             os.system('wget -P /home/t-lifen/ https://aka.ms/downloadazcopy-v10-linux')
#             os.system('tar -xvf /home/t-lifen/downloadazcopy-v10-linux -C /home/t-lifen/')
#         if dist.is_initialized():
#             dist.barrier()
            
#         for tsv in files:
#             if op.splitext(tsv)[1] == '.tsv':
#                 print('register tsv to create index', "tsv_id", tsv_id, tsv)
#                 lineidx = os.path.join(tsv_file, op.splitext(tsv)[0] + '.lineidx')
#                 line_name = op.splitext(tsv)[0] + '.lineidx'
#                 if azcopy and comm.is_main_process():
#                     if os.path.exists(os.path.join(_root_local, line_name)):
#                         print('file exists, skip azcopy')
#                     else:
#                         # azcopy data for main process
#                         print("azcopy file ", line_name)
#                         os.system(f'/home/t-lifen/azcopy_linux_amd64_10.19.0/./azcopy copy "https://vlpdatasets.blob.core.windows.net/data/lifeng/SAM-1B/tsv/{line_name}?sp=racwdli&st=2023-06-16T21:03:27Z&se=2023-12-02T06:03:27Z&spr=https&sv=2022-11-02&sr=c&sig=oaAjYEQ3fsr1JVZ8WNkGWGNv3%2F%2BDIo6279lvsbQBLFo%3D" "{_root_local}" --recursive')
#                         print("azcopy file ", tsv)
#                         os.system(f'/home/t-lifen/azcopy_linux_amd64_10.19.0/./azcopy copy "https://vlpdatasets.blob.core.windows.net/data/lifeng/SAM-1B/tsv/{tsv}?sp=racwdli&st=2023-06-16T21:03:27Z&se=2023-12-02T06:03:27Z&spr=https&sv=2022-11-02&sr=c&sig=oaAjYEQ3fsr1JVZ8WNkGWGNv3%2F%2BDIo6279lvsbQBLFo%3D" "{_root_local}" --recursive')
#                 if dist.is_initialized():
#                     dist.barrier()
#                 tsv_id += 1


def register_sam_instances(name, metadata, tsv_file):
    assert isinstance(name, str), name
    # assert isinstance(tsv_file, (str, os.PathLike)), tsv_file
    # azcopy_sam_tsv(tsv_file)

    DatasetCatalog.register(name, lambda: load_sam_index(tsv_file, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        tsv_file=tsv_file, evaluator_type="sam_interactive",  **metadata
    )


def register_all_sam_instance(root):
    for key, tsv_file in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_sam_instances(
            key,
            _get_sam_instances_meta(),
            os.path.join(root, tsv_file[0]),
        )

# def azcopy_data()

_root = os.getenv("SAM_DATASETS", "datasets")
_root_local = os.getenv("SAM_LOCAL", "no")
if _root_local != 'no':
    assert _root_local == '/home/t-lifen/sam_tsv'
    # assert os.path.exists(_root_local)
    # _root = _root_local
    # os.system(f'mkdir -p {_root_local}')
# else:
print("run register_all_sam_instance")
register_all_sam_instance(_root)
