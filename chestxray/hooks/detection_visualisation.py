import os.path as osp
from typing import Optional, Sequence

import mmcv
from mmdet.engine.hooks.visualization_hook import DetVisualizationHook as _DetVisualizationHook
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from mmengine.fileio import get
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist


@HOOKS.register_module(force=True)
class DetVisualizationHook(_DetVisualizationHook):

    def epoch2iter(self, interval: int, runner: Runner) -> int:
        """Convert epoch interval to iteration interval.

        Args:
            interval (int): The interval in epochs.
            runner (:obj:`Runner`): The runner of the training process.

        Returns:
            int: The interval in iterations.
        """
        return interval * len(runner._val_dataloader)

    def after_val_iter(
        self, runner: Runner, batch_idx: int, data_batch: dict, outputs: Sequence[DetDataSample]
    ) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        self.interval = self.epoch2iter(self.interval, runner)

        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order="rgb")

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else "val_img",
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter,
            )

    def after_test_iter(
        self, runner: Runner, batch_idx: int, data_batch: dict, outputs: Sequence[DetDataSample]
    ) -> None:
        pass
