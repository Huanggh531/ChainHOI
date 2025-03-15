from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import numpy as np
import torch
import lightning.pytorch as L
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import json
import os

from rich import get_console
from rich.table import Table

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def print_table(title, metrics):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    # dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    dist = torch.norm(activation[:, first_dices] - activation[:, second_dices], p=2, dim=2)
    return dist.mean()



@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    torch.set_float32_matmul_precision('high')

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    from src.models.metrics.compute import ComputeMetrics,ComputeMetrics_obj
    #metric_compute=ComputeMetrics(njoints=22,jointstype="humanml3d")
    metric_compute2=ComputeMetrics_obj(njoints=22,jointstype="humanml3d")
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
    
    # keys_list = list(state_dict.keys())
    # for key in keys_list:
    #     if 'orig_mod.' in key:
    #         deal_key = key.replace('_orig_mod.', '')
    #         state_dict[deal_key] = state_dict[key]
    #         del state_dict[key]
    model.load_state_dict(state_dict)

    log.info("Starting testing motion skeleton!")
    all_metrics = {}
    replication_times = cfg.model.metrics.replicate_times
    trainer.test(model, datamodule=datamodule)[0]
    from src.data.humanml.scripts.motion_process import recover_from_ric
    
    for i in range(len(model.gt_motion)):
        
        pred_motion_all=model.pred_motion[i]
        #pred_motion=pred_motion_all[...,0:263]
        #pred_motion=recover_from_ric(pred_motion.float(),22)
        gt_motion_all=model.gt_motion[i]
        #gt_motion=gt_motion_all[...,0:263]
        #gt_motion=recover_from_ric(gt_motion.float(),22)

        pred_obj=pred_motion_all[...,263:]
        gt_obj=gt_motion_all[...,263:]
        lengths=model.length[i]
        obj_points=model.obj_points[i]
        obj_name=model.obj_name[i]
        #metric_compute.update(pred_motion,gt_motion,lengths.cpu())
        metric_compute2.update(pred_obj,gt_obj,obj_points,obj_name,lengths.cpu())
    output=metric_compute2.compute(sanity_flag=False)
    print(output)
    import pdb
    pdb.set_trace()


    

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
