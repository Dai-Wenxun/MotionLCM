import os
import importlib
from typing import Type, TypeVar
from argparse import ArgumentParser

from omegaconf import OmegaConf, DictConfig


def get_module_config(cfg_model: DictConfig, path: str = "modules") -> DictConfig:
    files = os.listdir(f'./configs/{path}/')
    for file in files:
        if file.endswith('.yaml'):
            with open(f'./configs/{path}/' + file, 'r') as f:
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
    parser.add_argument("--cfg", type=str, required=True, help="config file")

    # Demo Args
    parser.add_argument('--example', type=str, required=False, help="input text and lengths with txt format")
    parser.add_argument('--no-plot', action="store_true", required=False, help="whether plot the skeleton-based motion")
    parser.add_argument('--replication', type=int, default=1, help="the number of replication of sampling")
    parser.add_argument('--vis', type=str, default="tb", choices=['tb', 'swanlab'], help="the visualization method, tensorboard or swanlab")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    cfg_model = get_module_config(cfg.model, cfg.model.target)
    cfg = OmegaConf.merge(cfg, cfg_model)

    cfg.example = args.example
    cfg.no_plot = args.no_plot
    cfg.replication = args.replication
    cfg.vis = args.vis
    return cfg
