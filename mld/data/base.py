import copy
from os.path import join as pjoin
from typing import Any, Callable

from torch.utils.data import DataLoader


class BaseDataModule:
    def __init__(self, collate_fn: Callable) -> None:
        super(BaseDataModule, self).__init__()
        self.collate_fn = collate_fn
        self.is_mm = False

    def get_sample_set(self, overrides: dict) -> Any:
        sample_params = copy.deepcopy(self.hparams)
        sample_params.update(overrides)
        split_file = pjoin(
            eval(f"self.cfg.DATASET.{self.name.upper()}.ROOT"),
            self.cfg.TEST.SPLIT + ".txt"
        )
        return self.Dataset(split_file=split_file, **sample_params)

    def __getattr__(self, item: str) -> Any:
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")].upper()
            item_c = "_" + item
            if item_c not in self.__dict__:
                split_file = pjoin(
                    eval(f"self.cfg.DATASET.{self.name.upper()}.ROOT"),
                    eval(f"self.cfg.{subset}.SPLIT") + ".txt"
                )
                self.__dict__[item_c] = self.Dataset(split_file=split_file, **self.hparams)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")

    def get_dataloader_options(self, stage: str) -> dict:
        stage_args = eval(f"self.cfg.{stage.upper()}")
        dataloader_options = {
            "batch_size": stage_args.BATCH_SIZE,
            "num_workers": stage_args.NUM_WORKERS,
            "collate_fn": self.collate_fn,
            "persistent_workers": stage_args.PERSISTENT_WORKERS,
        }
        return dataloader_options

    def train_dataloader(self) -> DataLoader:
        dataloader_options = self.get_dataloader_options('TRAIN')
        return DataLoader(self.train_dataset, shuffle=True, **dataloader_options)

    def val_dataloader(self) -> DataLoader:
        dataloader_options = self.get_dataloader_options('VAL')
        return DataLoader(self.val_dataset, shuffle=False, **dataloader_options)

    def test_dataloader(self) -> DataLoader:
        dataloader_options = self.get_dataloader_options('TEST')
        dataloader_options["batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        return DataLoader(self.test_dataset, shuffle=False, **dataloader_options)
