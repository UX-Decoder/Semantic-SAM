# Copyright (c) Microsoft, Inc. and its affiliates.
# Modified by Xueyan Zou and Jianwei Yang.
import json
import os
import collections
import glob
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_SAM_RAW = {
    "sam_train": (
        "meta_sa",
        (901,910)
    ),
    "sam_minitrain": (
        "meta_sa",
        (0,12)
    ),
    "sam_val": (
        "meta_sa",
        (901,902)
    ),
    "sam_minival": (
        "meta_sa",
        (998,999)
    ),    
}


def load_sam_instances(name: str, dirname: str, id_range: tuple):
    """
    Load SAM detection annotations to Detectron2 format.

    Args:
        name: name of split
        dirname: dataset directory path
        id_range: (start, end) tuple for dataset subfolders
    """
    dicts = []
    for id in range(*id_range):
        subfolder = os.path.join(dirname, 'sa_%06d' % id, 'image_list.da')
        dicts += torch.load(subfolder)
    return dicts

def register_sam(name, dirname, id_range):
    DatasetCatalog.register("{}".format(name), lambda: load_sam_instances(name, dirname, id_range))
    MetadataCatalog.get("{}".format(name)).set(
        dirname=dirname,
        thing_dataset_id_to_contiguous_id={},
    )

def register_all_sam(root):
    for (
        prefix,
        (image_root, id_range),
    ) in _PREDEFINED_SPLITS_SAM_RAW.items():
        register_sam(
            prefix,
            os.path.join(root, image_root),
            id_range
        )

_root = os.getenv("SAM_JSON", "datasets")
register_all_sam(_root)