# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json

PART_IN_CATEGORIES = [{'id': 0, 'name': 'Quadruped Head', 'supercategory': 'Quadruped'},
                      {'id': 1, 'name': 'Quadruped Body', 'supercategory': 'Quadruped'},
                      {'id': 2, 'name': 'Quadruped Foot', 'supercategory': 'Quadruped'},
                      {'id': 3, 'name': 'Quadruped Tail', 'supercategory': 'Quadruped'},
                      {'id': 4, 'name': 'Biped Head', 'supercategory': 'Biped'},
                      {'id': 5, 'name': 'Biped Body', 'supercategory': 'Biped'},
                      {'id': 6, 'name': 'Biped Hand', 'supercategory': 'Biped'},
                      {'id': 7, 'name': 'Biped Foot', 'supercategory': 'Biped'},
                      {'id': 8, 'name': 'Biped Tail', 'supercategory': 'Biped'},
                      {'id': 9, 'name': 'Fish Head', 'supercategory': 'Fish'},
                      {'id': 10, 'name': 'Fish Body', 'supercategory': 'Fish'},
                      {'id': 11, 'name': 'Fish Fin', 'supercategory': 'Fish'},
                      {'id': 12, 'name': 'Fish Tail', 'supercategory': 'Fish'},
                      {'id': 13, 'name': 'Bird Head', 'supercategory': 'Bird'},
                      {'id': 14, 'name': 'Bird Body', 'supercategory': 'Bird'},
                      {'id': 15, 'name': 'Bird Wing', 'supercategory': 'Bird'},
                      {'id': 16, 'name': 'Bird Foot', 'supercategory': 'Bird'},
                      {'id': 17, 'name': 'Bird Tail', 'supercategory': 'Bird'},
                      {'id': 18, 'name': 'Snake Head', 'supercategory': 'Snake'},
                      {'id': 19, 'name': 'Snake Body', 'supercategory': 'Snake'},
                      {'id': 20, 'name': 'Reptile Head', 'supercategory': 'Reptile'},
                      {'id': 21, 'name': 'Reptile Body', 'supercategory': 'Reptile'},
                      {'id': 22, 'name': 'Reptile Foot', 'supercategory': 'Reptile'},
                      {'id': 23, 'name': 'Reptile Tail', 'supercategory': 'Reptile'},
                      {'id': 24, 'name': 'Car Body', 'supercategory': 'Car'},
                      {'id': 25, 'name': 'Car Tier', 'supercategory': 'Car'},
                      {'id': 26, 'name': 'Car Side Mirror', 'supercategory': 'Car'},
                      {'id': 27, 'name': 'Bicycle Body', 'supercategory': 'Bicycle'},
                      {'id': 28, 'name': 'Bicycle Head', 'supercategory': 'Bicycle'},
                      {'id': 29, 'name': 'Bicycle Seat', 'supercategory': 'Bicycle'},
                      {'id': 30, 'name': 'Bicycle Tier', 'supercategory': 'Bicycle'},
                      {'id': 31, 'name': 'Boat Body', 'supercategory': 'Boat'},
                      {'id': 32, 'name': 'Boat Sail', 'supercategory': 'Boat'},
                      {'id': 33, 'name': 'Aeroplane Head', 'supercategory': 'Aeroplane'},
                      {'id': 34, 'name': 'Aeroplane Body', 'supercategory': 'Aeroplane'},
                      {'id': 35, 'name': 'Aeroplane Engine', 'supercategory': 'Aeroplane'},
                      {'id': 36, 'name': 'Aeroplane Wing', 'supercategory': 'Aeroplane'},
                      {'id': 37, 'name': 'Aeroplane Tail', 'supercategory': 'Aeroplane'},
                      {'id': 38, 'name': 'Bottle Mouth', 'supercategory': 'Bottle'},
                      {'id': 39, 'name': 'Bottle Body', 'supercategory': 'Bottle'}]


def _get_partimagenet_metadata(key):
    # if '_base' in key:
    #     id_to_name = {x['id']: x['name'] for x in PASCAL_PART_BASE_CATEGORIES}
    # else:
    id_to_name = {x['id']: x['name'] for x in PART_IN_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]

    part_classes = [a.split(" ")[1].lower() for a in thing_classes]
    thing_clases_id_to_part_id = {v: sorted(set(part_classes)).index(n) for v, n in enumerate(part_classes)}
    whole_classes = [a.split(" ")[0].lower() for a in thing_classes]
    thing_clases_id_to_whole_id = {v: sorted(set(whole_classes)).index(n) for v, n in enumerate(whole_classes)}
    thing_clases_id_to_flattened_wholepart = {tid: thing_clases_id_to_whole_id[tid] * len(set(part_classes)) + pid for
                                              tid, pid in thing_clases_id_to_part_id.items()}
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_clases_id_to_part_id": thing_clases_id_to_part_id,
        "part_classes": sorted(set(part_classes)),
        "thing_clases_id_to_whole_id": thing_clases_id_to_whole_id,
        "whole_classes": sorted(set(whole_classes)),
        "thing_clases_id_to_flattened_wholepart": thing_clases_id_to_flattened_wholepart,
    }


def register_partimagenet_part_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="pascal_part_interactive", **metadata
    )


_PART_IN = {
    "partimagenet_train": ("imagenet/train", "partimagenet/train_format.json"),
    # "pascal_part_train_one": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_one.json"),
    "partimagenet_val_inter": ("imagenet/val", "partimagenet/val_format_mini.json"),
   }


def register_partimagenet_part(root):
    for key, (image_root, json_file) in _PART_IN.items():
        register_partimagenet_part_instances(
            key,
            _get_partimagenet_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("PART_IN", "datasets")
register_partimagenet_part(_root)
