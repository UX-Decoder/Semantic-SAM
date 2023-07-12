from detectron2.data.datasets import get_lvis_instances_meta, register_lvis_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from xy_utils.lvis_cat import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
import logging
import os
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer
import json



_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_minival": ("coco/", "coco/annotations/lvis_v1_minival_inserted_image_name.json"),
        # "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        # "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        # "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        # "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    # "lvis_v0.5": {
    #     "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
    #     "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
    #     "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
    #     "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    # },
    # "lvis_v0.5_cocofied": {
    #     "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
    #     "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    # },
}

def get_lvis_instances_meta_v1():
    assert len(LVIS_V1_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    thing_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    # lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    # thing_classes = [k["name"] for k in O365_CATEGORIES]
    def preprocess_name(name):
        name = name.lower().strip()
        name = name.replace('_', ' ')
        return name
    thing_classes = [preprocess_name(k["synonyms"][0]) for k in LVIS_V1_CATEGORIES]
    meta = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
            }
    return meta


def register_lvis_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_lvis_json(image_root, json_file, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="lvis", **metadata
    )


def load_lvis_json(image_root, annot_json, metadata):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    with PathManager.open(annot_json) as f:
        json_info = json.load(f)

    imageid2seg = {}
    imageid2box = {}
    imageid2lable = {}
    for anno in json_info["annotations"]:
        image_id = anno['image_id']
        seg = anno["segmentation"]
        bbox = anno["bbox"]
        label = anno["category_id"]
        if image_id not in imageid2seg:
            imageid2seg[image_id] = []
        if image_id not in imageid2box:
            imageid2box[image_id] = []
        if image_id not in imageid2lable:
            imageid2lable[image_id] = []
        imageid2seg[image_id] += [seg]
        imageid2box[image_id] += [bbox]
        imageid2lable[image_id] += [label]

    ret = []
    cnt_empty = 0
    for image in json_info["images"]:
        image_file = os.path.join(image_root ,'/'.join(image["coco_url"].split('/')[-2:]))
        image_id = image['id']
        if image_id not in imageid2lable:
            cnt_empty += 1
            continue
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "height": image['height'],
                "width": image['width'],
                "instance": imageid2seg[image_id],
                "box": imageid2box[image_id],
                "labels": imageid2lable[image_id],
            }
        )

    print("Empty annotations: {}".format(cnt_empty))
    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta_v1(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


_root = os.getenv("DATASET3", "datasets")
register_all_lvis(_root)