import os
import importlib
from typing import Type, TypeVar
from argparse import ArgumentParser

from omegaconf import OmegaConf, DictConfig


def get_module_config(cfg_model: DictConfig, paths: list[str]) -> DictConfig:
    files = [os.path.join('./configs/modules', p+'.yaml') for p in paths]
    for file in files:
        assert os.path.exists(file), f'{file} is not exists.'
        with open(file, 'r') as f:
            cfg_model.merge_with(OmegaConf.load(f))
    return cfg_model


def get_obj_from_str(string: str, reload: bool = False) -> Type:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: DictConfig) -> TypeVar:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def parse_args() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="The main config file")
    parser.add_argument('--example', type=str, required=False, help="The input texts and lengths with txt format")
    parser.add_argument('--no-plot', action="store_true", required=False, help="Whether to plot the skeleton-based motion")
    parser.add_argument('--replication', type=int, default=1, help="The number of replications of sampling")
    parser.add_argument('--vis', type=str, default="tb", choices=['tb', 'swanlab'], help="The visualization backends: tensorboard or swanlab")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    cfg_model = get_module_config(cfg.model, cfg.model.target)
    cfg = OmegaConf.merge(cfg, cfg_model)

    cfg.example = args.example
    cfg.no_plot = args.no_plot
    cfg.replication = args.replication
    cfg.vis = args.vis
    return cfg
