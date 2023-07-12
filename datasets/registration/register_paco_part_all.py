# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import copy
# from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json
import json
_root = os.getenv("PACO", "datasets")
json_name = 'os.path.join(_root,"paco/annotations/paco_lvis_v1_val.json")'
if os.path.exists(json_name):
    with open(os.path.join(_root,"paco/annotations/paco_lvis_v1_val.json")) as f:
        j=json.load(f)
    PACO_CATEGORIES=j['categories']


def _get_paco_metadata(key):
    # if '_base' in key:
    #     id_to_name = {x['id']: x['name'] for x in PASCAL_PART_BASE_CATEGORIES}
    # else:
    id_to_name = {x['id']: x['name'] for x in PACO_CATEGORIES}

    thing_classes_ = [id_to_name[k] for k in sorted(id_to_name)]
    PACO_CATEGORIES_=copy.deepcopy(PACO_CATEGORIES)
    for cat in PACO_CATEGORIES_:
        if ':' not in cat['name']:
            cat['name']=cat['name']+':whole'
        if '_(' in cat['name']:
            cat['name']=cat['name'][:cat['name'].find('_(')]+cat['name'][cat['name'].find(')')+1:]
        if '_' in cat['name']:
            cat['name']=cat['name'].replace('_',' ')
    id_to_name = {x['id']: x['name'] for x in PACO_CATEGORIES_}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]

    part_classes = [a.split(":")[1].lower() for a in thing_classes]
    thing_clases_id_to_part_id={v: sorted(set(part_classes)).index(n) for v, n in enumerate(part_classes)}
    whole_classes = [a.split(":")[0].lower() for a in thing_classes]

    no_part_index = sorted(set(part_classes)).index('whole')
    thing_classes_id_without_part = [k for k, v in thing_clases_id_to_part_id.items() if no_part_index==v]

    thing_clases_id_to_whole_id={v: sorted(set(whole_classes)).index(n) for v, n in enumerate(whole_classes)}
    thing_clases_id_to_flattened_wholepart = {tid: thing_clases_id_to_whole_id[tid]*len(set(part_classes))+pid for tid, pid in thing_clases_id_to_part_id.items()}
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes_,
        "thing_clases_id_to_part_id": thing_clases_id_to_part_id,
        "part_classes": sorted(set(part_classes)),
        "thing_clases_id_to_whole_id": thing_clases_id_to_whole_id,
        "whole_classes": sorted(set(whole_classes)),
        "thing_clases_id_to_flattened_wholepart": thing_clases_id_to_flattened_wholepart,
        "thing_classes_id_without_part": thing_classes_id_without_part,
        }


def register_paco_part_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="pascal_part_interactive", **metadata
    )

_PACO = {
    "paco_train": ("coco", "paco/annotations/paco_lvis_v1_train.json"),
    # "pascal_part_train_one": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_one.json"),
    "paco_val_inter": ("coco", "paco/annotations/paco_lvis_v1_val_mini.json"),
    # "paco_test": ("paco/val2017", "paco/annotations/paco_lvis_v1_val.json"),
    # "pascal_part_base_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_base.json"),
    # "pascal_part_base_train_one": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_base_one.json"),
    # "imagenet_voc_parsed": ("imagenet/train", "imagenet/imagenet_voc_image_parsed.json"),
    # "imagenet_golden_pascal_parsed": ("imagenet/train", "imagenet/imagenet_golden_pascal_parsed.json"),
    # "imagenet_golden_pascal_parsed_swinbase": ("imagenet/train", "imagenet/imagenet_golden_pascal_parsed_swinbase.json"),
}


def register_paco_part(root):
    for key, (image_root, json_file) in _PACO.items():
        register_paco_part_instances(
            key,
            _get_paco_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

if os.path.exists(json_name):
    register_paco_part(_root)
