# ------------------------------------------------------------------------
# Copyright (c) MicroSoft, Inc. and its affiliates.
# Modified from OpenSeed https://github.com/IDEA-Research/OpenSeed by Feng Li (fliay@connect.ust.hk).
# ------------------------------------------------------------------------
"""
Semantic-SAM training and inference script. based on MaskDINO and OpenSeed.
"""
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import time

from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig, instantiate

# dataloader and evaluator
from datasets import (
    build_train_dataloader,
    build_evaluator,
    build_eval_dataloader,
)
import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
import weakref

from semantic_sam import build_model
from semantic_sam.BaseModel import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg['OUTPUT_DIR'],
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg['SOLVER']['MAX_ITER']
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        # TODO: release model conversion checkpointer from DINO to MaskDINO
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg['OUTPUT_DIR'],
            **kwargs,
        )
        # TODO: release GPU cluster submit scripts based on submitit for multi-node training

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = copy.deepcopy(self.cfg)
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=1))
        return ret

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = BaseModel(cfg, build_model(cfg)).cuda()
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_train_dataloader(cfg, )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # import ipdb; ipdb.set_trace()
        loader = build_eval_dataloader(cfg, )
        return loader

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
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
                        if comm.is_main_process():
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

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = copy.deepcopy(cfg)
        # frozen = cfg.is_frozen()
        # cfg.defrost()

        assert (
                cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )
        return cfg

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        from utils.misc import hook_metadata, hook_switcher, hook_opt
        from detectron2.utils.logger import log_every_n_seconds
        import datetime
        # build dataloade
        dataloaders = cls.build_test_loader(cfg, dataset_name=None)
        dataset_names = cfg['DATASETS']['TEST']
        model = model.eval().cuda()
        model_without_ddp = model
        if not type(model) == BaseModel:
            model_without_ddp = model.module

        for dataloader, dataset_name in zip(dataloaders, dataset_names):
            # build evaluator
            evaluator = build_evaluator(cfg, dataset_name, cfg['OUTPUT_DIR'])
            evaluator.reset()
            with torch.no_grad():
                # setup task
                if 'sam' in dataset_names:
                    task = 'multi_granularity'
                else:
                    task = 'interactive'

                hook_switcher(model_without_ddp, dataset_name)
                # setup timer
                total = len(dataloader)
                num_warmup = min(5, total - 1)
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0
                start_data_time = time.perf_counter()

                for idx, batch in enumerate(dataloader):
                    total_data_time += time.perf_counter() - start_data_time
                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0
                        total_eval_time = 0
                    start_compute_time = time.perf_counter()

                    # forward
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        # import ipdb; ipdb.set_trace()
                        outputs = model(batch, inference_task=task)

                    total_compute_time += time.perf_counter() - start_compute_time
                    start_eval_time = time.perf_counter()

                    evaluator.process(batch, outputs)
                    total_eval_time += time.perf_counter() - start_eval_time

                    iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                    data_seconds_per_iter = total_data_time / iters_after_start
                    compute_seconds_per_iter = total_compute_time / iters_after_start
                    eval_seconds_per_iter = total_eval_time / iters_after_start
                    total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

                    if comm.is_main_process() and (idx >= num_warmup * 2 or compute_seconds_per_iter > 5):
                        eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                        log_every_n_seconds(
                            logging.INFO,
                            (
                                f"Inference done {idx + 1}/{total}. "
                                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                f"ETA={eta}"
                            ),
                            n=5,
                        )
                    start_data_time = time.perf_counter()

            # evaluate
            results = evaluator.evaluate()
        model = model.train().cuda()

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def main(args=None):
    cfg = setup(args)
    print("Command cfg:", cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    trainer = Trainer(cfg)
    if len(args.lang_weight)>0:
        # load language weight for semantic
        import copy
        weight = copy.deepcopy(trainer.cfg.MODEL.WEIGHTS)
        trainer.cfg.MODEL.WEIGHTS = args.lang_weight
        print("load original language language weight!!!!!!")
        trainer.resume_or_load(resume=args.resume)
        trainer.cfg.MODEL.WEIGHTS = weight
    print("load pretrained model weight!!!!!!")
    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()
                
if __name__ == "__main__":
    # main()
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    parser.add_argument('--lang_weight', type=str, default='')
    args = parser.parse_args()
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
