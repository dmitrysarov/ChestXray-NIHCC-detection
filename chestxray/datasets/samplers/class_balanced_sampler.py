import math
from abc import ABC
from collections import defaultdict

import torch
from mmcls.datasets import BaseDataset
from mmcls_model.datasets.builder import SAMPLERS
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler


class BaseBalancedDistributedSampler(_DistributedSampler, ABC):
    def __init__(
        self,
        dataset: BaseDataset,
        num_replicas: int,
        rank: int,
        total_size: int,
        num_epochs: int,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.num_samples = int(math.ceil(total_size) * 1.0 / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.num_epochs = num_epochs
        self.instance_cls_counts = self._calculate_instance_cls_counts(dataset)

    @staticmethod
    def _calculate_instance_cls_counts(dataset):
        cls2count = defaultdict(int)
        classes = []
        num_images = len(dataset)
        for idx in range(num_images):
            assert len(dataset.get_cat_ids(idx)) == 1
            cls = int(dataset.get_cat_ids(idx)[0])
            cls2count[cls] += 1
            classes.append(cls)
        return torch.tensor([cls2count[cls] for cls in classes], dtype=torch.float)

    def _calculate_instance_weights(self):
        raise NotImplementedError

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # randomly select classes
        weights = self._calculate_instance_weights()
        random_indices = torch.multinomial(
            weights, num_samples=self.total_size, generator=g, replacement=True
        ).tolist()

        assert len(random_indices) == self.total_size

        # subsample
        indices = random_indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch + 1  # otherwise it starts from 0


@SAMPLERS.register_module()
class InstanceBalancedDistributedSampler(BaseBalancedDistributedSampler):
    def _calculate_instance_weights(self):
        return torch.ones_like(self.instance_cls_counts)


@SAMPLERS.register_module()
class ClassBalancedDistributedSampler(BaseBalancedDistributedSampler):
    def _calculate_instance_weights(self):
        return 1 / self.instance_cls_counts


@SAMPLERS.register_module()
class WeightedProgressivelyBalancedDistributedSampler(BaseBalancedDistributedSampler):
    def _calculate_instance_weights(self):
        w = self.epoch / self.num_epochs
        class_balanced_weights = 1 / self.instance_cls_counts
        instance_balanced_weights = torch.ones_like(self.instance_cls_counts)

        class_balanced_weights /= class_balanced_weights.sum()
        instance_balanced_weights /= instance_balanced_weights.sum()

        return class_balanced_weights * w + instance_balanced_weights * (1 - w)


@SAMPLERS.register_module()
class PowerProgressivelyBalancedDistributedSampler(BaseBalancedDistributedSampler):
    def _calculate_instance_weights(self):
        q = self.epoch / self.num_epochs
        return 1 / torch.pow(self.instance_cls_counts, q)


class BaseBalancedSampler(Sampler, ABC):
    def __init__(
        self,
        dataset: BaseDataset,
        total_size: int,
        num_epochs: int,
    ):
        self.num_samples = total_size
        self.total_size = total_size
        self.num_epochs = num_epochs
        self.instance_cls_counts = self._calculate_instance_cls_counts(dataset)
        self.epoch = 0

    @staticmethod
    def _calculate_instance_cls_counts(dataset):
        cls2count = defaultdict(int)
        classes = []
        num_images = len(dataset)
        for idx in range(num_images):
            assert len(dataset.get_cat_ids(idx)) == 1
            cls = int(dataset.get_cat_ids(idx)[0])
            cls2count[cls] += 1
            classes.append(cls)
        return torch.tensor([cls2count[cls] for cls in classes], dtype=torch.float)

    def _calculate_instance_weights(self):
        raise NotImplementedError

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # randomly select classes
        weights = self._calculate_instance_weights()
        random_indices = torch.multinomial(
            weights, num_samples=self.total_size, generator=g, replacement=True
        ).tolist()

        assert len(random_indices) == self.total_size

        # subsample
        indices = random_indices
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch + 1  # otherwise it starts from 0


@SAMPLERS.register_module()
class InstanceBalancedSampler(BaseBalancedSampler):
    def _calculate_instance_weights(self):
        return torch.ones_like(self.instance_cls_counts)


@SAMPLERS.register_module()
class ClassBalancedSampler(BaseBalancedSampler):
    def _calculate_instance_weights(self):
        return 1 / self.instance_cls_counts


@SAMPLERS.register_module()
class WeightedProgressivelyBalancedSampler(BaseBalancedSampler):
    def _calculate_instance_weights(self):
        w = self.epoch / self.num_epochs
        class_balanced_weights = 1 / self.instance_cls_counts
        instance_balanced_weights = torch.ones_like(self.instance_cls_counts)

        class_balanced_weights /= class_balanced_weights.sum()
        instance_balanced_weights /= instance_balanced_weights.sum()

        return class_balanced_weights * w + instance_balanced_weights * (1 - w)


@SAMPLERS.register_module()
class PowerProgressivelyBalancedSampler(BaseBalancedSampler):
    def _calculate_instance_weights(self):
        q = self.epoch / self.num_epochs
        return 1 / torch.pow(self.instance_cls_counts, q)
