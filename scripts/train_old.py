# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import traceback

import mlflow
import mmcv
import torch
from mmcls import __version__
from mmcls.apis import set_random_seed
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger
from mmcls_model.apis.train import train_model
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist


def check_git_clarity():
    from git import Repo

    repo = Repo(".")
    assert repo.is_dirty() == False, "Repository contain unstaged/uncommited changes, please commit before train"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--no-validate", action="store_true", help="whether not to evaluate the checkpoint during training"
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument("--device", help="device used for training")
    group_gpus.add_argument(
        "--gpus", type=int, help="number of gpus to use " "(only applicable to non-distributed training)"
    )
    group_gpus.add_argument(
        "--gpu-ids", type=int, nargs="+", help="ids of gpus to use " "(only applicable to non-distributed training)"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic", action="store_true", help="whether to set deterministic options for CUDNN backend."
    )
    parser.add_argument("--options", nargs="+", action=DictAction, help="arguments in dict")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="exp")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    if not os.environ.get("DEBUG", False):
        check_git_clarity()
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # add run name
    if cfg.get("custom_hooks", None):
        for i, _ in enumerate(cfg["custom_hooks"]):
            if cfg["custom_hooks"][i].get("type", None) == "MLflowHook":
                cfg["custom_hooks"][i]["run_name"] = args.run_name

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["exp_name"] = osp.basename(args.config)

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # mlog.init(
    #     run_name=osp.basename(args.config),
    #     tags=dict(host=os.environ["HOSTNAME"], **{k.replace(",", " "): str(v) for k, v in env_info_dict.items()}),
    # )

    # set random seeds
    if args.seed is not None:
        logger.info(f"Set random seed to {args.seed}, " f"deterministic: {args.deterministic}")
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed

    try:
        model = build_classifier(cfg.model)
        model.init_weights()
        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmcls version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmcls_version=__version__, config=cfg.pretty_text, CLASSES=datasets[0].CLASSES
            )
        # add an attribute for visualization convenience
        train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            device="cpu" if args.device == "cpu" else "cuda",
            meta=meta,
        )
    except Exception as err:
        # ml = mlog.get(__name__)
        mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
        mlflow.end_run("FAILED")
        # ml.fail()
        raise err


if __name__ == "__main__":
    main()
