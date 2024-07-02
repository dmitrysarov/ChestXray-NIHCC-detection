# Copyright (c) OpenMMLab. All rights reserved.


import argparse
import os
import os.path as osp
import sys
import traceback
from pathlib import Path

import torch
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

import chestxray  # to set up registry
from chestxray.hooks.mlflow import MLflowHook

__all__ = ["chestxray"]


def check_git_clarity():
    from git import Repo

    repo = Repo(".")
    assert repo.is_dirty() == False, "Repository contain unstaged/uncommited changes, please commit before train"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--run-name", help="name of experiment")
    parser.add_argument("--amp", action="store_true", default=False, help="enable automatic-mixed-precision training")
    parser.add_argument("--auto-scale-lr", action="store_true", help="enable automatically scaling LR.")
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        help="If specify checkpoint path, resume from it, while if not "
        "specify, try to auto resume from the latest checkpoint "
        "in the work directory.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    try:
        if not os.environ.get("DEBUG", False):
            check_git_clarity()
        args = parse_args()
        os.environ["MLFLOW_TRACKING_URI"] = "http://ec2-54-194-129-138.eu-west-1.compute.amazonaws.com:5000/"
        # os.environ["MLFLOW_EXPERIMENT_NAME"] = "chestxray"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        # Reduce the number of repeated compilations and improve
        # training speed.
        setup_cache_size_limit_of_dynamo()

        # load config
        cfg = Config.fromfile(args.config)

        cfg.launcher = args.launcher
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        # work_dir is determined in this priority: CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get("work_dir", None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0], args.run_name)

        if cfg.get("custom_hooks", None):
            for i, _ in enumerate(cfg["custom_hooks"]):
                if cfg["custom_hooks"][i].get("type", None) == "MLflowHook":
                    cfg["custom_hooks"][i]["run_name"] = args.run_name

        if cfg.get("default_hooks", None):
            for key, value in cfg["default_hooks"].items():
                if cfg["default_hooks"][key].get("type", None) == "DetVisualizationHook":
                    cfg["default_hooks"][key]["test_out_dir"] = "prediction_images"

        # enable automatic-mixed-precision training
        if args.amp is True:
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

        # enable automatically scaling LR
        if args.auto_scale_lr:
            if "auto_scale_lr" in cfg and "enable" in cfg.auto_scale_lr and "base_batch_size" in cfg.auto_scale_lr:
                cfg.auto_scale_lr.enable = True
            else:
                raise RuntimeError(
                    'Can not find "auto_scale_lr" or '
                    '"auto_scale_lr.enable" or '
                    '"auto_scale_lr.base_batch_size" in your'
                    " configuration file."
                )
        cfg.load_from = "yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
        # resume is determined in this priority: resume from > auto_resume
        if args.resume == "auto":
            cfg.resume = True
            cfg.load_from = None
        elif args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume

        # build the runner from config
        if "runner_type" not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        def inspect_gradient(grad):
            # print(grad)  # print gradient
            if torch.isnan(grad).any():
                print("NaN value found in gradient. Stopping training...")

        # Register hook for each parameter in your model
        for param in runner.model.parameters():
            param.register_hook(inspect_gradient)

            # start training
        # runner.train()
        # runner.load_checkpoint(str(Path(runner.work_dir) / "best.pth"))
        runner.call_hook("before_run")
        runner._has_loaded = False
        runner._resume = False
        runner._load_from = "/kaggle/working/ChestXray-NIHCC-detection/work_dirs/yolox_tiny_8xb8-300e_coco_notebook/coco_pretrained/best.pth"
        runner.load_or_resume()
        print("runner loaded checkpoint")
        runner.test()
    except Exception as e:
        mlflow_hook = [hook for hook in runner.hooks if isinstance(hook, MLflowHook)][0]
        mlflow_hook.ml.log_text(traceback.format_exc(), "error_traceback.txt")
        mlflow_hook.ml.log_artifact(
            (Path(runner.log_dir) / Path(runner.log_dir).name).with_suffix(".log"), artifact_path=""
        )
        mlflow_hook.ml.end_run("FAILED")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
