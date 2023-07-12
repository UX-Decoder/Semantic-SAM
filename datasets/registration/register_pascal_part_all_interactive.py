# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json

PASCAL_PART_CATEGORIES = [
    {"id": 1, "name": "aeroplane:body"},
    {"id": 2, "name": "aeroplane:wing"},
    {"id": 3, "name": "aeroplane:tail"},
    {"id": 4, "name": "aeroplane:wheel"},
    {"id": 5, "name": "bicycle:wheel"},
    {"id": 6, "name": "bicycle:handlebar"},
    {"id": 7, "name": "bicycle:saddle"},
    {"id": 8, "name": "bird:beak"},
    {"id": 9, "name": "bird:head"},
    {"id": 10, "name": "bird:eye"},
    {"id": 11, "name": "bird:leg"},
    {"id": 12, "name": "bird:foot"},
    {"id": 13, "name": "bird:wing"},
    {"id": 14, "name": "bird:neck"},
    {"id": 15, "name": "bird:tail"},
    {"id": 16, "name": "bird:torso"},
    {"id": 17, "name": "bottle:body"},
    {"id": 18, "name": "bottle:cap"},
    {"id": 19, "name": "bus:license plate", "abbr": "bus:liplate"},
    {"id": 20, "name": "bus:headlight"},
    {"id": 21, "name": "bus:door"},
    {"id": 22, "name": "bus:mirror"},
    {"id": 23, "name": "bus:window"},
    {"id": 24, "name": "bus:wheel"},
    {"id": 25, "name": "car:license plate", "abbr": "car:liplate"},
    {"id": 26, "name": "car:headlight"},
    {"id": 27, "name": "car:door"},
    {"id": 28, "name": "car:mirror"},
    {"id": 29, "name": "car:window"},
    {"id": 30, "name": "car:wheel"},
    {"id": 31, "name": "cat:head"},
    {"id": 32, "name": "cat:leg"},
    {"id": 33, "name": "cat:ear"},
    {"id": 34, "name": "cat:eye"},
    {"id": 35, "name": "cat:paw", "abbr": "cat:pa"},
    {"id": 36, "name": "cat:neck"},
    {"id": 37, "name": "cat:nose"},
    {"id": 38, "name": "cat:tail"},
    {"id": 39, "name": "cat:torso"},
    {"id": 40, "name": "cow:head"},
    {"id": 41, "name": "cow:leg"},
    {"id": 42, "name": "cow:ear"},
    {"id": 43, "name": "cow:eye"},
    {"id": 44, "name": "cow:neck"},
    {"id": 45, "name": "cow:horn"},
    {"id": 46, "name": "cow:muzzle"},
    {"id": 47, "name": "cow:tail"},
    {"id": 48, "name": "cow:torso"},
    {"id": 49, "name": "dog:head"},
    {"id": 50, "name": "dog:leg"},
    {"id": 51, "name": "dog:ear"},
    {"id": 52, "name": "dog:eye"},
    {"id": 53, "name": "dog:paw", "abbr": "dog:pa"},
    {"id": 54, "name": "dog:neck"},
    {"id": 55, "name": "dog:nose"},
    {"id": 56, "name": "dog:muzzle"},
    {"id": 57, "name": "dog:tail"},
    {"id": 58, "name": "dog:torso"},
    {"id": 59, "name": "horse:head"},
    {"id": 60, "name": "horse:leg"},
    {"id": 61, "name": "horse:ear"},
    {"id": 62, "name": "horse:eye"},
    {"id": 63, "name": "horse:neck"},
    {"id": 64, "name": "horse:muzzle"},
    {"id": 65, "name": "horse:tail"},
    {"id": 66, "name": "horse:torso"},
    {"id": 67, "name": "motorbike:wheel"},
    {"id": 68, "name": "motorbike:handlebar"},
    {"id": 69, "name": "motorbike:headlight"},
    {"id": 70, "name": "motorbike:saddle"},
    {"id": 71, "name": "person:hair"},
    {"id": 72, "name": "person:head"},
    {"id": 73, "name": "person:ear"},
    {"id": 74, "name": "person:eye"},
    {"id": 75, "name": "person:nose"},
    {"id": 76, "name": "person:neck"},
    {"id": 77, "name": "person:mouth"},
    {"id": 78, "name": "person:arm"},
    {"id": 79, "name": "person:hand"},
    {"id": 80, "name": "person:leg"},
    {"id": 81, "name": "person:foot"},
    {"id": 82, "name": "person:torso"},
    {"id": 83, "name": "potted plant:plant"},
    {"id": 84, "name": "potted plant:pot"},
    {"id": 85, "name": "sheep:head"},
    {"id": 86, "name": "sheep:leg"},
    {"id": 87, "name": "sheep:ear"},
    {"id": 88, "name": "sheep:eye"},
    {"id": 89, "name": "sheep:neck"},
    {"id": 90, "name": "sheep:horn"},
    {"id": 91, "name": "sheep:muzzle"},
    {"id": 92, "name": "sheep:tail"},
    {"id": 93, "name": "sheep:torso"},
]


PASCAL_PART_BASE_CATEGORIES = [
    {"id": 1, "name": "aeroplane:body"},
    {"id": 2, "name": "aeroplane:wing"},
    {"id": 3, "name": "aeroplane:tail"},
    {"id": 4, "name": "aeroplane:wheel"},
    {"id": 5, "name": "bicycle:wheel"},
    {"id": 6, "name": "bicycle:handlebar"},
    {"id": 7, "name": "bicycle:saddle"},
    {"id": 8, "name": "bird:beak"},
    {"id": 9, "name": "bird:head"},
    {"id": 10, "name": "bird:eye"},
    {"id": 11, "name": "bird:leg"},
    {"id": 12, "name": "bird:foot"},
    {"id": 13, "name": "bird:wing"},
    {"id": 14, "name": "bird:neck"},
    {"id": 15, "name": "bird:tail"},
    {"id": 16, "name": "bird:torso"},
    {"id": 17, "name": "bottle:body"},
    {"id": 18, "name": "bottle:cap"},
    {"id": 19, "name": "car:license plate", "abbr": "car:liplate"},
    {"id": 20, "name": "car:headlight"},
    {"id": 21, "name": "car:door"},
    {"id": 22, "name": "car:mirror"},
    {"id": 23, "name": "car:window"},
    {"id": 24, "name": "car:wheel"},
    {"id": 25, "name": "cat:head"},
    {"id": 26, "name": "cat:leg"},
    {"id": 27, "name": "cat:ear"},
    {"id": 28, "name": "cat:eye"},
    {"id": 29, "name": "cat:paw", "abbr": "cat:pa"},
    {"id": 30, "name": "cat:neck"},
    {"id": 31, "name": "cat:nose"},
    {"id": 32, "name": "cat:tail"},
    {"id": 33, "name": "cat:torso"},
    {"id": 34, "name": "cow:head"},
    {"id": 35, "name": "cow:leg"},
    {"id": 36, "name": "cow:ear"},
    {"id": 37, "name": "cow:eye"},
    {"id": 38, "name": "cow:neck"},
    {"id": 39, "name": "cow:horn"},
    {"id": 40, "name": "cow:muzzle"},
    {"id": 41, "name": "cow:tail"},
    {"id": 42, "name": "cow:torso"},
    {"id": 43, "name": "horse:head"},
    {"id": 44, "name": "horse:leg"},
    {"id": 45, "name": "horse:ear"},
    {"id": 46, "name": "horse:eye"},
    {"id": 47, "name": "horse:neck"},
    {"id": 48, "name": "horse:muzzle"},
    {"id": 49, "name": "horse:tail"},
    {"id": 50, "name": "horse:torso"},
    {"id": 51, "name": "motorbike:wheel"},
    {"id": 52, "name": "motorbike:handlebar"},
    {"id": 53, "name": "motorbike:headlight"},
    {"id": 54, "name": "motorbike:saddle"},
    {"id": 55, "name": "person:hair"},
    {"id": 56, "name": "person:head"},
    {"id": 57, "name": "person:ear"},
    {"id": 58, "name": "person:eye"},
    {"id": 59, "name": "person:nose"},
    {"id": 60, "name": "person:neck"},
    {"id": 61, "name": "person:mouth"},
    {"id": 62, "name": "person:arm"},
    {"id": 63, "name": "person:hand"},
    {"id": 64, "name": "person:leg"},
    {"id": 65, "name": "person:foot"},
    {"id": 66, "name": "person:torso"},
    {"id": 67, "name": "potted plant:plant"},
    {"id": 68, "name": "potted plant:pot"},
    {"id": 69, "name": "sheep:head"},
    {"id": 70, "name": "sheep:leg"},
    {"id": 71, "name": "sheep:ear"},
    {"id": 72, "name": "sheep:eye"},
    {"id": 73, "name": "sheep:neck"},
    {"id": 74, "name": "sheep:horn"},
    {"id": 75, "name": "sheep:muzzle"},
    {"id": 76, "name": "sheep:tail"},
    {"id": 77, "name": "sheep:torso"},
]


PASCAL_PART_NOVEL_CATEGORIES = [
    {"id": 1, "name": "bus:license plate", "abbr": "bus:liplate"},
    {"id": 2, "name": "bus:headlight"},
    {"id": 3, "name": "bus:door"},
    {"id": 4, "name": "bus:mirror"},
    {"id": 5, "name": "bus:window"},
    {"id": 6, "name": "bus:wheel"},
    {"id": 7, "name": "dog:head"},
    {"id": 8, "name": "dog:leg"},
    {"id": 9, "name": "dog:ear"},
    {"id": 10, "name": "dog:eye"},
    {"id": 11, "name": "dog:paw", "abbr": "dog:pa"},
    {"id": 12, "name": "dog:neck"},
    {"id": 13, "name": "dog:nose"},
    {"id": 14, "name": "dog:muzzle"},
    {"id": 15, "name": "dog:tail"},
    {"id": 16, "name": "dog:torso"},
]


def _get_partimagenet_metadata(key):
    if '_base' in key:
        id_to_name = {x['id']: x['name'] for x in PASCAL_PART_BASE_CATEGORIES}
    else:
        id_to_name = {x['id']: x['name'] for x in PASCAL_PART_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    
    
    part_classes = [a.split(":")[1] for a in thing_classes]
    thing_clases_id_to_part_id={v: sorted(set(part_classes)).index(n) for v, n in enumerate(part_classes)}
    whole_classes = [a.split(":")[0] for a in thing_classes]
    thing_clases_id_to_whole_id={v: sorted(set(whole_classes)).index(n) for v, n in enumerate(whole_classes)}
    thing_clases_id_to_flattened_wholepart = {tid: thing_clases_id_to_whole_id[tid]*len(set(part_classes))+pid for tid, pid in thing_clases_id_to_part_id.items()}
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_clases_id_to_part_id": thing_clases_id_to_part_id,
        "part_classes": sorted(set(part_classes)),
        "thing_clases_id_to_whole_id": thing_clases_id_to_whole_id,
        "whole_classes": sorted(set(whole_classes)),
        "thing_clases_id_to_flattened_wholepart": thing_clases_id_to_flattened_wholepart,
        }


def register_pascal_part_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="pascal_part_interactive", **metadata
    )

_PASCAL_PART = {
    # "pascal_part_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train.json"),
    # "pascal_part_train_one": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_one.json"),
    "pascal_part_val_interactive": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/val.json"),
    # "pascal_part_base_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_base.json"),
    # "pascal_part_base_train_one": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_base_one.json"),
    # "imagenet_voc_parsed": ("imagenet/train", "imagenet/imagenet_voc_image_parsed.json"),
    # "imagenet_golden_pascal_parsed": ("imagenet/train", "imagenet/imagenet_golden_pascal_parsed.json"),
    # "imagenet_golden_pascal_parsed_swinbase": ("imagenet/train", "imagenet/imagenet_golden_pascal_parsed_swinbase.json"),
}


def register_pascal_part(root):
    for key, (image_root, json_file) in _PASCAL_PART.items():
        register_pascal_part_instances(
            key,
            _get_partimagenet_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("PASCAL", "datasets")
register_pascal_part(_root)
