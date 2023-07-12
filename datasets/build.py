# Copyright (c) Facebook, Inc. and its affiliates.
import os
import itertools
import logging
import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
from detectron2.data.build import (
    build_batch_data_loader,
    load_proposals_into_dataset,
    trivial_batch_collator,
)
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
from fvcore.common.config import CfgNode
from omegaconf import DictConfig, OmegaConf

from .dataset_mappers import (
    SamBaselineDatasetMapperJSON,
    SamBaselineDatasetMapper,
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    ImageNetDatasetMapper,
    O365InstanceNewBaselineDatasetMapper,
    DatasetMapperFilterByBox,
    PascalInstanceNewBaselineDatasetMapper,
    COCOInteractivePanopticNewBaselineDatasetMapper,
    CoCoInferenceDatasetMapper,
    PartFilterWholeInstanceNewBaselineDatasetMapper,
    MaskFormerPanopticDatasetMapperInteractive,
)
from .evaluation import (InstanceSegEvaluator,
                         SemSegEvaluator,
                         COCOPanopticEvaluator,
                         PASCALPARTEvaluator,
                         InteractiveEvaluator,
                         JointBoxPointInteractiveEvaluator,
)
from semantic_sam.utils import configurable
from detectron2.utils.comm import get_world_size, is_main_process
from typing import Any, Dict, List, Set

class JointLoader(torchdata.IterableDataset):
    def __init__(self, loaders, key_dataset):
        dataset_names = []
        for key, loader in loaders.items():
            name = "{}".format(key.split('_')[0])
            setattr(self, name, loader)
            dataset_names += [name]
        self.dataset_names = dataset_names
        self.key_dataset = key_dataset
    
    def __iter__(self):
        for batch in zip(*[getattr(self, name) for name in self.dataset_names]):
            yield {key: batch[i] for i, key in enumerate(self.dataset_names)}

    def __len__(self):
        return len(getattr(self, self.key_dataset))

def filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if isinstance(ann, list):
                for instance in ann:
                    if instance.get("iscrowd", 0) == 0:
                        return True
            else:
                if ann.get("iscrowd", 0) == 0:
                    return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def get_detection_dataset_dicts(
    dataset_names, filter_empty=True, proposal_files=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names)

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=None,
    )
    # import ipdb;ipdb.set_trace()
    if mapper is None:
        if isinstance(cfg, (DictConfig)):
            cfg = OmegaConf.to_container(copy.deepcopy(cfg))
        mapper_cfg = CfgNode({'INPUT': cfg['INPUT'], 'MODEL': cfg['MODEL'], 'DATASETS': cfg['DATASETS']})
        mapper = DatasetMapper(mapper_cfg, False)
    assert cfg['TEST']['BATCH_SIZE_TOTAL'] % get_world_size() == 0, "Evaluation total batchsize is not divisible by gpu number"
    batch_size = cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size()

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg['DATALOADER']['NUM_WORKERS'],
        "sampler": InferenceSampler(len(dataset)),
        "batch_size": batch_size,
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def _train_loader_from_config(cfg, dataset_name, mapper, *, dataset=None, sampler=None):
    cfg_datasets = cfg['DATASETS']
    cfg_dataloader = cfg['DATALOADER']
    
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg_dataloader['FILTER_EMPTY_ANNOTATIONS'],
            proposal_files=cfg_datasets['PROPOSAL_FILES_TRAIN'] if cfg_dataloader['LOAD_PROPOSALS'] else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg_dataloader['SAMPLER_TRAIN']
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        sampler = TrainingSampler(len(dataset))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg['TRAIN']['BATCH_SIZE_TOTAL'],
        "aspect_ratio_grouping": cfg_dataloader['ASPECT_RATIO_GROUPING'],
        "num_workers": cfg_dataloader['NUM_WORKERS'],
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


def get_config_from_name(cfg, dataset_name):
    # adjust config according to dataset
    # joint mapper for multiple part data
    joint_part = cfg['DATASETS'].get('JOINT_PART_LOADER', False)
    if 'sam' in dataset_name:
        cfg.update(cfg['SAM'])
        return cfg
    if joint_part and ('pascal' in dataset_name or 'paco' in dataset_name or 'partimagenet' in dataset_name):
        cfg.update(cfg['PART_ALL'])
        return cfg
    elif 'pascal' in dataset_name:
        cfg.update(cfg['PSACAL_PART'])
        return cfg
    elif 'refcoco' in dataset_name:
        cfg.update(cfg['REF'])
        return cfg
    elif 'coco' in dataset_name:
        if 'COCO' in cfg.keys():
            cfg.update(cfg['COCO'])
        return cfg
    elif 'ade' in dataset_name:
        if 'ADE20K' in cfg.keys():
            cfg.update(cfg['ADE20K'])
        return cfg
    elif 'imagenet' in dataset_name:
        if 'IMAGENET' in cfg.keys():
            cfg.update(cfg['IMAGENET'])
        return cfg
    elif 'vlp' in dataset_name:
        cfg.update(cfg['VLP'])
        return cfg
    elif 'sun' in dataset_name:
        cfg.update(cfg['SUN'])
        return cfg
    elif 'object365' in dataset_name:
        cfg.update(cfg['OBJECT365'])
        return cfg
    elif 'scan' in dataset_name:
        cfg.update(cfg['SCAN'])
        return cfg
    elif 'cityscape' in dataset_name:
        cfg.update(cfg['CITY'])
        return cfg
    elif 'bdd' in dataset_name:
        cfg.update(cfg['BDD'])
        return cfg
    else:
        assert False, "dataset not support."


def build_eval_dataloader(cfg, ):
    dataloaders = []
    cfg = copy.deepcopy(cfg)
    for dataset_name in cfg['DATASETS']['TEST']:
        cfg = get_config_from_name(cfg, dataset_name)
        # adjust mapper according to dataset
        if dataset_name == 'imagenet_val':
            mapper = ImageNetDatasetMapper(cfg, False)
        elif 'coco' in dataset_name:
            mapper = CoCoInferenceDatasetMapper(cfg, False)
        elif dataset_name == 'sam_val':
            mapper = SamBaselineDatasetMapper(cfg, False)
        elif dataset_name == 'sam_minival':
            mapper = SamBaselineDatasetMapperJSON(cfg, False)
        else:
            mapper = None
        if dataset_name == 'pascal_part_val_interactive':
            mapper = PascalInstanceNewBaselineDatasetMapper(cfg, True)
            # dataloaders += [build_custom_test_loader(cfg, dataset_name, mapper=mapper)]
            dataloaders += [build_detection_test_loader(cfg, dataset_name, mapper=mapper)]
        else:
            dataloaders += [build_detection_test_loader(cfg, dataset_name, mapper=mapper)]
    return dataloaders


def build_train_dataloader(cfg, ):
    dataset_names = cfg['DATASETS']['TRAIN']
    
    loaders = {}
    cfg = copy.deepcopy(cfg)
    for dataset_name in dataset_names:
        cfg = get_config_from_name(cfg, dataset_name)
        mapper_name = cfg['INPUT']['DATASET_MAPPER_NAME']
        # Sam data dataset mapper
        if mapper_name == "sam":
            # print("using SamBaselineDatasetMapperJSON")
            # mapper = SamBaselineDatasetMapperJSON(cfg, True)
            mapper = SamBaselineDatasetMapper(cfg, True)
            loaders['sam'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "pascal_part":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            loaders['pascal'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # pascal_part dataset mapper
        elif mapper_name == "pascal_part_lsj":
            mapper = PascalInstanceNewBaselineDatasetMapper(cfg, True)
            loaders['pascal'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # shared part dataset mapper
        elif mapper_name == "part_all_filter_whole_lsj":
            mapper = PartFilterWholeInstanceNewBaselineDatasetMapper(cfg, True,dataset_name=dataset_name)
            # FIXME shared loader for all part data
            name = dataset_name.split('_')[0]
            loaders[name] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # Semantic segmentation dataset mapper
        elif mapper_name == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            loaders['coco'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif mapper_name == "mask_former_panoptic":   # TODO: Hack for ade training; should add ade name
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            loaders['ade'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # Instance segmentation dataset mapper
        elif mapper_name == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            loaders['coco'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif mapper_name == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            loaders['coco'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif mapper_name == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            loaders['coco'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "coco_interactive_panoptic_lsj":  # FIXME filter empty instances
            mapper = COCOInteractivePanopticNewBaselineDatasetMapper(cfg, True)
            loaders['coco'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "mask_former_panoptic_interactive":  # FIXME filter empty instances
            mapper = MaskFormerPanopticDatasetMapperInteractive(cfg, True, dataset_name=dataset_name)
            loaders['ade'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "object365":
            mapper = O365InstanceNewBaselineDatasetMapper(cfg, True)  # Use lsj instance mapper for o365
            loaders['o365'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        else:
            mapper = None
            loaders[dataset_name] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
    # import ipdb; ipdb.set_trace()
    if len(loaders) == 1 and not cfg['LOADER'].get('JOINT', False):
        for k, v in loaders.items():
            print("number of iterations per epoch: ", v, len(loaders[k]))
        return list(loaders.values())[0]
        # return loaders.values()['coco']
        # return loaders['coco']
    else:
        return JointLoader(loaders, key_dataset=cfg['LOADER'].get('KEY_DATASET', 'coco'))

    
def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """

    cfg_model_decoder_test = cfg["MODEL"]["DECODER"]["TEST"]

    if output_folder is None:
        output_folder = os.path.join(cfg["OUTPUT_DIR"], "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # for pascal part
    if evaluator_type == "pascal_part":
        return PASCALPARTEvaluator(dataset_name, output_dir=output_folder)
    # FIXME interactive
    if evaluator_type in ['sam_interactive', 'pascal_part_interactive', 'coco_panoptic_seg_interactive']:
        evaluator_list.append(InteractiveEvaluator(dataset_name, output_dir=output_folder))
    # for box interactive evaluation
    if evaluator_type in ['coco_panoptic_seg_interactive_jointboxpoint']:
        box_interactive = cfg_model_decoder_test.get('BOX_INTERACTIVE', False)
        evaluator_list.append(JointBoxPointInteractiveEvaluator(dataset_name, output_dir=output_folder, box_interactive=box_interactive))
        # evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    if evaluator_type == 'sam':
        evaluator_list.append(COCOEvaluator("coco_2017_val", output_dir=output_folder))

    # semantic segmentation
    if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    # instance segmentation
    if evaluator_type == "coco":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

    # panoptic segmentation
    if evaluator_type in [
        "coco_panoptic_seg",
        # "coco_panoptic_seg_interactive",
        "ade20k_panoptic_seg",
        "cityscapes_panoptic_seg",
        "mapillary_vistas_panoptic_seg",
        "scannet_panoptic_seg",
        "bdd_panoptic_pano"
    ]:
        if cfg_model_decoder_test["PANOPTIC_ON"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    # COCO
    if (evaluator_type == "coco_panoptic_seg" and cfg_model_decoder_test["INSTANCE_ON"]) or evaluator_type == "object365_od":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if (evaluator_type == "coco_panoptic_seg" and cfg_model_decoder_test["SEMANTIC_ON"]) or evaluator_type == "coco_sem_seg":
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    # Mapillary Vistas
    if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg_model_decoder_test["INSTANCE_ON"]:
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg_model_decoder_test["SEMANTIC_ON"]:
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    # Cityscapes
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "cityscapes_panoptic_seg":
        if cfg_model_decoder_test["SEMANTIC_ON"]:
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if cfg_model_decoder_test["INSTANCE_ON"]:
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
    # ADE20K
    if evaluator_type == "ade20k_panoptic_seg" and cfg_model_decoder_test["INSTANCE_ON"]:
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    # SEGINW
    if evaluator_type == "seginw" and cfg_model_decoder_test["INSTANCE_ON"]:
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    # LVIS
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]


    
    return DatasetEvaluators(evaluator_list)


def build_optimizer(cls, cfg, model):
    cfg_solver = cfg['SOLVER']
    weight_decay_norm = cfg_solver['WEIGHT_DECAY_NORM']
    weight_decay_embed = cfg_solver['WEIGHT_DECAY_EMBED']
    weight_decay_bias = cfg_solver.get('WEIGHT_DECAY_BIAS', 0.0)

    defaults = {}
    defaults["lr"] = cfg_solver['BASE_LR']
    defaults["weight_decay"] = cfg_solver['WEIGHT_DECAY']

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    lr_multiplier = cfg['SOLVER']['LR_MULTIPLIER']
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)

            for key, lr_mul in lr_multiplier.items():
                if key in "{}.{}".format(module_name, module_param_name):
                    hyperparams["lr"] = hyperparams["lr"] * lr_mul
                    if is_main_process():
                        logger.info("Modify Learning rate of {}: {}".format(
                            "{}.{}".format(module_name, module_param_name), lr_mul))

            if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
            ):
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            if "bias" in module_name:
                hyperparams["weight_decay"] = weight_decay_bias
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg_solver['CLIP_GRADIENTS']['CLIP_VALUE']
        enable = (
                cfg_solver['CLIP_GRADIENTS']['ENABLED']
                and cfg_solver['CLIP_GRADIENTS']['CLIP_TYPE'] == "full_model"
                and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg_solver['OPTIMIZER']
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg_solver['BASE_LR'], momentum=cfg_solver['MOMENTUM']
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg_solver['BASE_LR']
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    return optimizer