import copy
from typing import Callable, Optional

import numpy as np
from omegaconf import DictConfig

import torch

from .base import BaseDataModule
from .humanml.dataset import Text2MotionDatasetV2, MotionDataset
from .humanml.scripts.motion_process import recover_from_ric


class KitDataModule(BaseDataModule):

    def __init__(self,
                 cfg: DictConfig,
                 motion_only: bool,
                 batch_size: int,
                 num_workers: int,
                 collate_fn: Optional[Callable] = None,
                 persistent_workers: bool = True,
                 **kwargs) -> None:
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         persistent_workers=persistent_workers)
        self.hparams = copy.deepcopy(kwargs)
        self.name = 'kit'
        self.njoints = 21
        self.nfeats = 251
        self.cfg = cfg
        self.Dataset = MotionDataset if motion_only else Text2MotionDatasetV2
        sample_overrides = {"tiny": True, "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)

    def denorm_spatial(self, hint: torch.Tensor) -> torch.Tensor:
        raw_mean = torch.tensor(self._sample_set.raw_mean).to(hint)
        raw_std = torch.tensor(self._sample_set.raw_std).to(hint)
        hint = hint * raw_std + raw_mean
        return hint

    def norm_spatial(self, hint: torch.Tensor) -> torch.Tensor:
        raw_mean = torch.tensor(self._sample_set.raw_mean).to(hint)
        raw_std = torch.tensor(self._sample_set.raw_std).to(hint)
        hint = (hint - raw_mean) / raw_std
        return hint

    def feats2joints(self, features: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.hparams['mean']).to(features)
        std = torch.tensor(self.hparams['std']).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def renorm4t2m(self, features: torch.Tensor) -> torch.Tensor:
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams['mean']).to(features)
        ori_std = torch.tensor(self.hparams['std']).to(features)
        eval_mean = torch.tensor(self.hparams['mean_eval']).to(features)
        eval_std = torch.tensor(self.hparams['std_eval']).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on: bool = True) -> None:
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
