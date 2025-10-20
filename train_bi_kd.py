# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import argparse
import copy
import gc
import itertools
import logging
import os
import sys
import time

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from omegaconf import OmegaConf

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig, get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import EventStorage, CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

import pdb


def kldiv(predictions1, predictions2):
    kl_divs = []
    
    for pred1, pred2 in zip(predictions1, predictions2):
        # Get original tensor shape
        sem_seg1 = pred1['sem_seg']
        sem_seg2 = pred2['sem_seg']
        
        # Calculate number of chunks based on tensor size
        # Adjust chunk_size based on your GPU memory
        N = sem_seg1.shape[0]  # number of classes
        chunk_size = max(1, N // 8)  # Process 1/8th at a time
        num_chunks = (N + chunk_size - 1) // chunk_size
        
        chunk_kls = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, N)
            
            with torch.amp.autocast(device_type='cuda'):
                # Process chunk by chunk
                s1 = sem_seg1[start_idx:end_idx].float()
                s2 = sem_seg2[start_idx:end_idx].float()
                
                # Calculate KL div for this chunk
                log_prob1 = F.log_softmax(s1, dim=0)
                prob2 = F.softmax(s2, dim=0)
                
                # Free up memory immediately
                del s1, s2
                
                chunk_kl = F.kl_div(log_prob1, prob2, reduction='batchmean')
                chunk_kls.append(chunk_kl.detach())
                
                # Free up memory immediately
                del log_prob1, prob2
                torch.cuda.empty_cache()
        
        # Calculate mean KL divergence across chunks
        kl = torch.mean(torch.stack(chunk_kls))
        kl_divs.append(kl)
        
        # Clear chunk results
        del chunk_kls
        torch.cuda.empty_cache()
    
    result = torch.stack(kl_divs)
    return result



def custom_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--t_config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--s_config", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


# adapted from:
# https://github.com/pytorch/tnt/blob/ebda066f8f55af6a906807d35bc829686618074d/torchtnt/utils/device.py#L328-L346
def _set_float32_precision(precision: str = "high") -> None:
    """Sets the precision of float32 matrix multiplications and convolution operations.

    For more information, see the PyTorch docs:
    - https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    - https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32

    Args:
        precision: The setting to determine which datatypes to use for matrix
        multiplication and convolution operations.
    """
    if not (torch.cuda.is_available()):  # Not relevant for non-CUDA devices
        return
    # set precision for matrix multiplications
    torch.set_float32_matmul_precision(precision)
    # set precision for convolution operations
    if precision == "highest":
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True


def custom_setup(t_cfg, s_cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        t_cfg (CfgNode or omegaconf.DictConfig): the full teacher config to be used
        s_cfg (CfgNode or omegaconf.DictConfig): the full student config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(t_cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "t_config") and args.t_config != "":
        logger.info(
            "Contents of args.t_config={}:\n{}".format(
                args.t_config,
                _highlight(PathManager.open(args.t_config, "r").read(), args.t_config),
            )
        )

    if hasattr(args, "s_config") and args.s_config != "":
        logger.info(
            "Contents of args.s_config={}:\n{}".format(
                args.s_config,
                _highlight(PathManager.open(args.s_config, "r").read(), args.s_config),
            )
        )

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # t_config.yaml and s_config.yaml in output directory
        t_path = os.path.join(output_dir, "t_config.yaml")
        s_path = os.path.join(output_dir, "s_config.yaml")
        if isinstance(t_cfg, CfgNode):
            logger.info("Running with full teacher config:\n{}".format(_highlight(t_cfg.dump(), ".yaml")))
            with PathManager.open(t_path, "w") as f:
                f.write(t_cfg.dump())
        else:
            LazyConfig.save(t_cfg, t_path)
        logger.info("Full teacher config saved to {}".format(t_path))

        if isinstance(s_cfg, CfgNode):
            logger.info("Running with full student config:\n{}".format(_highlight(s_cfg.dump(), ".yaml")))
            with PathManager.open(s_path, "w") as f:
                f.write(s_cfg.dump())
        else:
            LazyConfig.save(s_cfg, s_path)
        logger.info("Full student config saved to {}".format(s_path))
        

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(t_cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            t_cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )

    fp32_precision = _try_get_key(t_cfg, "FLOAT32_PRECISION", "train.float32_precision", default="")
    if fp32_precision != "":
        logger.info(f"Set fp32 precision to {fp32_precision}")
        _set_float32_precision(fp32_precision)
        logger.info(f"{torch.get_float32_matmul_precision()=}")
        logger.info(f"{torch.backends.cuda.matmul.allow_tf32=}")
        logger.info(f"{torch.backends.cudnn.allow_tf32=}")


def custom_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    PathManager.mkdirs(output_dir)
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
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
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
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
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
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

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

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
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    t_cfg = get_cfg()
    s_cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(t_cfg)
    add_deeplab_config(s_cfg)
    add_maskformer2_config(t_cfg)
    add_maskformer2_config(s_cfg)
    # cfg.merge_from_file(args.config_file)
    # load teacher and student config
    t_cfg.merge_from_file(args.t_config)
    s_cfg.merge_from_file(args.s_config)
    t_args = args.opts[:2]
    s_args = args.opts[2:]
    t_cfg.merge_from_list(t_args)
    s_cfg.merge_from_list(s_args)
    t_cfg.freeze()
    s_cfg.freeze()
    custom_setup(t_cfg, s_cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=t_cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return t_cfg, s_cfg


def test(t_cfg, s_cfg, teacher_model, student_model):
    t_res = Trainer.test(t_cfg, teacher_model)
    if t_cfg.TEST.AUG.ENABLED:
        t_res.update(Trainer.test_with_TTA(t_cfg, teacher_model))
    if comm.is_main_process():
        verify_results(t_cfg, t_res)

    s_res = Trainer.test(s_cfg, student_model)
    if s_cfg.TEST.AUG.ENABLED:
        s_res.update(Trainer.test_with_TTA(s_cfg, student_model))
    if comm.is_main_process():
        verify_results(s_cfg, s_res)
    return t_res, s_res


def train(t_cfg, s_cfg, teacher_model, student_model, resume):
    teacher_model.train()
    student_model.train()

    best_t_mIoU = -np.inf
    best_s_mIoU = -np.inf
    
    teacher_optimizer = Trainer.build_optimizer(t_cfg, teacher_model)
    student_optimizer = Trainer.build_optimizer(s_cfg, student_model)

    teacher_scheduler = Trainer.build_lr_scheduler(t_cfg, teacher_optimizer)
    student_scheduler = Trainer.build_lr_scheduler(s_cfg, student_optimizer)

    assert t_cfg.OUTPUT_DIR == s_cfg.OUTPUT_DIR
    
    t_checkpointer = DetectionCheckpointer(
        teacher_model, t_cfg.OUTPUT_DIR, optimizer=teacher_optimizer, scheduler=teacher_scheduler
    )
    s_checkpointer = DetectionCheckpointer(
        student_model, s_cfg.OUTPUT_DIR, optimizer=student_optimizer, scheduler=student_scheduler
    )

    t_start_iter = (
        t_checkpointer.resume_or_load(t_cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    s_start_iter = (
        s_checkpointer.resume_or_load(s_cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

    assert t_start_iter == s_start_iter
    assert t_cfg.SOLVER.MAX_ITER == s_cfg.SOLVER.MAX_ITER

    max_iter = t_cfg.SOLVER.MAX_ITER

    mapper = MaskFormerSemanticDatasetMapper(t_cfg, True)
    data_loader = build_detection_train_loader(t_cfg, mapper=mapper)
    logger = logging.getLogger("detectron2.trainer")
    logger.info("Starting training from iteration {}".format(t_start_iter))
    tic = time.time()
    for data, iteration in zip(data_loader, range(t_start_iter, max_iter)):

        if iteration % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()


        teacher_optimizer.zero_grad()
        student_optimizer.zero_grad()

        t_loss_dict, t_pred = teacher_model(data)
        s_loss_dict, s_pred = student_model(data)

        t_pred = [{k: v.detach() for k, v in pred.items()} for pred in t_pred]
        s_pred = [{k: v.detach() for k, v in pred.items()} for pred in s_pred]

        t_losses = sum(t_loss_dict.values())
        s_losses = sum(s_loss_dict.values())

        assert torch.isfinite(t_losses).all(), t_loss_dict
        assert torch.isfinite(s_losses).all(), s_loss_dict
        
        mask_t = (t_losses < s_losses).float()
        mask_s = (t_losses >= s_losses).float()

        kl_t = kldiv(s_pred, t_pred)
        kl_s = kldiv(t_pred, s_pred)

        loss = (t_losses + s_losses + mask_t * kl_t + mask_s * kl_s).mean()

        loss.backward()
        teacher_optimizer.step()
        student_optimizer.step()
        
        teacher_scheduler.step()
        student_scheduler.step()

        assert t_cfg.TEST.EVAL_PERIOD == s_cfg.TEST.EVAL_PERIOD

        if (
            t_cfg.TEST.EVAL_PERIOD > 0
            and (iteration + 1) % t_cfg.TEST.EVAL_PERIOD == 0
            and iteration != max_iter - 1
        ):
            t_res, s_res = test(t_cfg, s_cfg, teacher_model, student_model)
            t_mIoU = t_res["sem_seg"]["mIoU"]
            s_mIoU = s_res["sem_seg"]["mIoU"]
            if t_mIoU > best_t_mIoU:
                best_t_mIoU = t_mIoU
                t_checkpointer.save("best_teacher", **t_res)
            if s_mIoU > best_s_mIoU:
                best_s_mIoU = s_mIoU
                s_checkpointer.save("best_student", **s_res)
            comm.synchronize()

        if iteration - t_start_iter > 5 and (
            (iteration + 1) % 20 == 0 or iteration == max_iter - 1
        ):
            print("Iteration: {:d}, t_loss: {:.3f}, s_loss: {:.3f}, kl_t: {:.3f}, kl_s: {:.3f}, minutes: {:.3f}".format(iteration, t_losses.mean().item(), s_losses.mean().item(), kl_t.mean().item(), kl_s.mean().item(), (time.time() - tic) / 60))


def main(args):
    t_cfg, s_cfg = setup(args)
    teacher_model = Trainer.build_model(t_cfg)
    student_model = Trainer.build_model(s_cfg)
    DetectionCheckpointer(teacher_model, save_dir=t_cfg.OUTPUT_DIR).resume_or_load(t_cfg.MODEL.WEIGHTS, resume=args.resume)
    DetectionCheckpointer(student_model, save_dir=s_cfg.OUTPUT_DIR).resume_or_load(s_cfg.MODEL.WEIGHTS, resume=args.resume)
    
    if args.eval_only:
        t_res = Trainer.test(t_cfg, teacher_model)
        if t_cfg.TEST.AUG.ENABLED:
            t_res.update(Trainer.test_with_TTA(t_cfg, teacher_model))
        if comm.is_main_process():
            verify_results(t_cfg, t_res)

        s_res = Trainer.test(s_cfg, student_model)
        if s_cfg.TEST.AUG.ENABLED:
            s_res.update(Trainer.test_with_TTA(s_cfg, student_model))
        if comm.is_main_process():
            verify_results(s_cfg, s_res)
        return t_res, s_res
    
    train(t_cfg, s_cfg, teacher_model, student_model, args.resume)
    return test(t_cfg, s_cfg, teacher_model, student_model)


if __name__ == "__main__":
    args = custom_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
    