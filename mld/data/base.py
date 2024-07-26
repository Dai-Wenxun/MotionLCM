import copy
from os.path import join as pjoin
from typing import Any, Callable

from torch.utils.data import DataLoader

from .humanml.dataset import Text2MotionDatasetV2


class BaseDataModule:
    def __init__(self, collate_fn: Callable, batch_size: int,
                 num_workers: int, persistent_workers: bool) -> None:
        super(BaseDataModule, self).__init__()
        self.dataloader_options = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "persistent_workers": persistent_workers
        }
        self.is_mm = False

    def get_sample_set(self, overrides: dict) -> Text2MotionDatasetV2:
        sample_params = copy.deepcopy(self.hparams)
        sample_params.update(overrides)
        split_file = pjoin(
            eval(f"self.cfg.DATASET.{self.name.upper()}.ROOT"),
            self.cfg.VAL.SPLIT + ".txt"
        )
        return self.Dataset(split_file=split_file, **sample_params)

    def __getattr__(self, item: str) -> Any:
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")].upper()
            item_c = "_" + item
            if item_c not in self.__dict__:
                split = eval(f"self.cfg.{subset}.SPLIT")
                split_file = pjoin(
                    eval(f"self.cfg.DATASET.{self.name.upper()}.ROOT"),
                    eval(f"self.cfg.{subset}.SPLIT") + ".txt"
                )
                self.__dict__[item_c] = self.Dataset(split_file=split_file, **self.hparams)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_options)

    def val_dataloader(self) -> DataLoader:
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.VAL.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.VAL.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(self.val_dataset, **dataloader_options)

    def test_dataloader(self) -> DataLoader:
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(self.test_dataset, **dataloader_options)
