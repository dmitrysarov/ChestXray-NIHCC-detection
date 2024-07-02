# Copyright (c) Open-MMLab. All rights reserved.
import json
import os
import os.path as osp
import re
import shutil
import time
from collections.abc import MutableMapping, MutableSequence
from functools import partial
from glob import glob
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from git import InvalidGitRepositoryError, Repo

# from loguru import logger
from mmengine import Config
from mmengine.dist.utils import master_only
from mmengine.fileio import FileClient, dump
from mmengine.hooks.logger_hook import DATA_BATCH, LoggerHook
from mmengine.registry import HOOKS

from chestxray.logger import get_logger

logger = get_logger(__name__, level="DEBUG")


def get_git_remote_url():
    from git import Repo

    repo = Repo(".")
    return next(repo.remotes[0].urls)


def get_model_norm(model: torch.nn.Module) -> float:
    weights_norm = 0.0
    for param in model.parameters():
        weights_norm += torch.norm(param)
    return weights_norm


def flatten(d: Union[MutableMapping, MutableMapping, Any], parent_key: str = "", sep: str = ".", ignore_keys=()):
    items = []
    if parent_key in ignore_keys:
        return {parent_key: "ignored"}
    if isinstance(d, MutableMapping):
        for k, v in d.items():
            new_key = parent_key + sep + str(k) if parent_key else k
            items.extend(flatten(v, new_key, sep=sep, ignore_keys=ignore_keys).items())
    elif isinstance(d, MutableSequence):
        need_to_flatten = any([isinstance(v, (MutableMapping, MutableSequence)) for v in d])

        if need_to_flatten:
            for i, v in enumerate(d):
                items.extend(flatten(v, f"{parent_key}.{i}", sep=sep, ignore_keys=ignore_keys).items())
        else:
            items.append((parent_key, d))

    else:
        items.append((parent_key, d))
    return dict(items)


def batchify_dict(data, batch_size=100):
    for i in range(0, len(data), batch_size):
        yield dict(list(data.items())[i : i + batch_size])


@HOOKS.register_module()
class MLflowHook(LoggerHook):
    def __init__(
        self,
        interval=10,
        ignore_last=False,
        reset_flag=False,
        by_epoch=True,
        log_model=True,
        log_model_interval=1,
        save_last=True,
        ignore_keys=(),
        run_name="exp",
    ):
        super().__init__(interval=interval, ignore_last=ignore_last)  # , reset_flag, by_epoch)
        self.log_model = log_model
        self.log_model_interval = log_model_interval
        self.save_last = save_last
        self.ignore_keys = ignore_keys
        self.run_name = run_name
        self.ml = mlflow
        self.run = None

    def log_min_max_metrics(self, runner):
        run_id = self.ml.active_run().info.run_id

        mlflow_client = mlflow.tracking.MlflowClient()
        minmax_log_metrics = {}
        for tag in mlflow_client.get_run(run_id).data.metrics:
            for func in [min, max]:
                minmax_log_metrics[f"{func.__name__}_{tag}"] = func(
                    [m.value for m in mlflow_client.get_metric_history(run_id, tag)]
                )
        self.ml.log_metrics(minmax_log_metrics, step=runner.iter + 1)

    def upload_artifacts_subproc(self, local_path: str, artifact_path: Optional[str] = None):
        partial_log = partial(self.ml.log_artifact, local_path, artifact_path, run_id=self.run_id)
        proc = Process(target=partial_log, args=(local_path, artifact_path))
        proc.start()

    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        self.ml.start_run(run_name=self.run_name)
        self.run_id = self.ml.active_run().info.run_id
        config_path = osp.join(osp.join(runner.log_dir, "vis_data", "config.py"))
        cfg = dict(Config.fromfile(config_path))

        # hack to avoid mlflow limit of 100 keys
        for batch_params in batchify_dict(flatten(cfg, ignore_keys=self.ignore_keys), batch_size=100):
            self.ml.log_params(batch_params)
        self.ml.log_params({"git_remote": get_git_remote_url()})
        self.ml.log_params({"DEBUG": int(os.environ.get("DEBUG", 0))})
        number_of_folds = os.environ.get("NUMBER_OF_FOLDS", None)
        if number_of_folds:
            self.ml.log_param("number_of_folds", int(number_of_folds))
        if os.path.exists("dvc.lock"):
            import hashlib

            self.ml.log_param("dvc_dataset_version", hashlib.md5(open("dvc.lock", "rb").read()).hexdigest())
        try:
            repo = Repo(".")
            self.ml.set_tag("git remote repo", repo.remote().url)
        except InvalidGitRepositoryError:
            print("no git repository")

        # save config as a file
        self.upload_artifacts_subproc(config_path, artifact_path="")

    @master_only
    def after_run(self, runner):
        if osp.exists(osp.join(runner.work_dir, "best_calibrated.pth")):
            self.upload_artifacts_subproc(
                osp.join(runner.work_dir, "best_calibrated.pth"),
                artifact_path="checkpoints",
            )
        self.log_min_max_metrics(runner)
        self.ml.log_artifacts(
            os.path.join(runner.work_dir, str(runner.timestamp), "prediction_images"),
            artifact_path="prediction_images",
            run_id=self.run_id,
        )
        super().after_run(runner)
        self.ml.end_run()

    @master_only
    def after_train_iter(
        self, runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: Optional[dict] = None
    ) -> None:
        """Record logs after training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        import ipdb

        ipdb.set_trace()

        tag, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, "train")
        self.ml.log_metrics(tag, step=runner.iter + 1)

    @master_only
    def after_test_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.test_dataloader), "test", with_non_scalar=True
        )
        self.ml.log_metrics(tag, step=runner.epoch + 1)

    @master_only
    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None):
        logger.debug("Logging metrics for val epoch")
        super(MLflowHook, self).after_val_epoch(runner)

        tag, log_str = runner.log_processor.get_log_after_epoch(runner, len(runner.val_dataloader), "val")
        self.ml.log_metrics(tag, step=runner.epoch)

        if self.log_model:

            if self.is_last_train_epoch(runner):
                best_checkpoints = sorted(
                    glob(osp.join(runner.work_dir, "best_*.pth")),
                    key=lambda x: int(re.findall("epoch_(\d+).pth", x)[0]),
                )

                if best_checkpoints:
                    best_chck = best_checkpoints[-1]
                    best_epoch = int(re.findall("epoch_(\d+).pth", best_chck)[0])
                    self.ml.log_metric("best_epoch_number", best_epoch)
                    best_unified_path = osp.join(runner.work_dir, "best.pth")
                    shutil.move(best_chck, best_unified_path)
                    self.upload_artifacts_subproc(
                        best_unified_path,
                        artifact_path="checkpoints",
                    )
                else:
                    logger.warning(
                        "No best model found. Either your metirc for `save_best` is not specified or ill-defined."
                    )
            if self.save_last:
                if self.is_last_train_epoch(runner):
                    self.upload_artifacts_subproc(
                        osp.join(runner.work_dir, "latest.pth"),
                        artifact_path="checkpoints",
                    )
            if self.every_n_epochs(runner, self.log_model_interval):
                import ipdb

                ipdb.set_trace()
                logger.debug("Upload model and images")
                self.upload_artifacts_subproc(
                    osp.join(runner.work_dir, f"epoch_{runner.epoch}.pth"),
                    artifact_path="checkpoints",
                )
                for img_file in (Path(runner.log_dir) / "vis_data" / "vis_image").rglob("*.png"):
                    self.ml.log_artifact(str(img_file), artifact_path="pred_annotation", run_id=self.run_id)
                    os.remove(str(img_file))
