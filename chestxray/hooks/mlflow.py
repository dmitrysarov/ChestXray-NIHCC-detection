# Copyright (c) Open-MMLab. All rights reserved.
import json
import os
import os.path as osp
import re
import shutil
from collections.abc import MutableMapping, MutableSequence
from glob import glob
from multiprocessing import Process
from pathlib import Path
from typing import Any, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from git import InvalidGitRepositoryError, Repo

# from loguru import logger
from mmengine import Config
from mmengine.dist.utils import master_only
from mmengine.hooks import LoggerHook
from mmengine.registry import HOOKS

from chestxray.logger import get_logger

logger = get_logger(__name__)


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
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
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
        print(minmax_log_metrics)
        print(self.get_iter(runner))
        self.ml.log_metrics(minmax_log_metrics, step=self.get_iter(runner))

    def upload_artifacts_subproc(self, local_path: str, artifact_path: Optional[str] = None):
        proc = Process(target=self.ml.log_artifact, args=(local_path, artifact_path))
        proc.start()

    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        self.ml.start_run(run_name=self.run_name)
        cfg = dict(Config.fromfile(osp.join(runner.work_dir, runner.meta["exp_name"])))

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
        config_path = osp.join(runner.work_dir, runner.meta["exp_name"])
        unified_config_path = osp.join(runner.work_dir, "config.py")
        shutil.copyfile(config_path, unified_config_path)
        self.upload_artifacts_subproc(unified_config_path, artifact_path="")
        if os.path.exists("meta"):
            self.upload_artifacts_subproc("meta", artifact_path="")
        if os.path.exists("mmcls_model"):
            self.upload_artifacts_subproc("mmcls_model", artifact_path="")
        if os.path.exists("mmcls_model"):
            self.upload_artifacts_subproc("scripts", artifact_path="")

        # save session_id with were used for training
        # dvc_yaml_path = Path(os.environ["PROJECT_ROOT"]) / "dvc.yaml"
        # with open(dvc_yaml_path, "r") as f:
        #     yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
        # path = "/tmp"
        # df = pd.read_parquet(
        #     {k: v for item in yaml_data["vars"] for k, v in item.items()}["output_annotation"]["authentic"]
        # )
        # pd.DataFrame(df["session_id"].unique(), columns=["session_id"]).to_csv(
        #     Path(path) / "session_id.csv", index=False
        # )
        # self.upload_artifacts_subproc(str(Path(path) / "session_id.csv"), artifact_path="")

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            # rename val/0_ and val/1_ to val/val_ and val/subtrain_
            for key in list(tags):
                for old_key, new_key in zip(["val/0_", "val/1_"], ["val/val_", "val/subtrain_"]):
                    if old_key in key:
                        tags[key.replace(old_key, new_key)] = tags.pop(key)
            # this part will be saved to mlflow as json files
            complex_value_tags = {}
            tensor_value_tags = {}
            keys = list(tags.keys())
            for tag in keys:
                if isinstance(tags[tag], (MutableMapping, MutableSequence)):
                    complex_value_tags[tag] = tags[tag]
                    tags.pop(tag)
                    continue
                if isinstance(tags[tag], (torch.Tensor,)):
                    tensor_value_tags[tag] = tags[tag]
                    tags.pop(tag)
            try:
                for tag, val in complex_value_tags.items():
                    output_path = osp.join(runner.work_dir, f"{tag.replace('/', '_')}_{runner.epoch + 1:04d}.json")
                    with open(output_path, "w") as f:
                        json.dump(val, f)
                    self.upload_artifacts_subproc(output_path, artifact_path="classification_reports")
                for tag, val in tensor_value_tags.items():
                    output_path = osp.join(runner.work_dir, f"{tag.replace('/', '_')}_{runner.epoch + 1:04d}.txt")
                    np.savetxt(output_path, val.numpy(), fmt="%.2f")
                    self.upload_artifacts_subproc(output_path, artifact_path="confusion_matrix")
            except Exception as e:
                logger.error(e)
            # and this part contains just metrics
            try:
                weights_norm = get_model_norm(runner.model)
                tags.update({"weights_norm": weights_norm})
                self.ml.log_metrics(tags, step=self.get_epoch(runner))
            except Exception as e:
                logger.error(e)

    @master_only
    def after_run(self, runner):
        if osp.exists(osp.join(runner.work_dir, "best_calibrated.pth")):
            self.upload_artifacts_subproc(
                osp.join(runner.work_dir, "best_calibrated.pth"),
                artifact_path="checkpoints",
            )
        self.log_min_max_metrics(runner)
        super().after_run(runner)
        self.ml.end_run()

    @master_only
    def after_train_epoch(self, runner):
        super(MLflowHook, self).after_train_epoch(runner)

        if self.log_model:
            if self.is_last_epoch(runner):
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
                if self.is_last_epoch(runner):
                    self.upload_artifacts_subproc(
                        osp.join(runner.work_dir, "latest.pth"),
                        artifact_path="checkpoints",
                    )
            if self.every_n_epochs(runner, self.log_model_interval):
                self.upload_artifacts_subproc(
                    osp.join(runner.work_dir, f"epoch_{runner.epoch + 1}.pth"), artifact_path="checkpoints"
                )
