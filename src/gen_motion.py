from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import numpy as np
import torch
import lightning.pytorch as L
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from data.humanml.scripts.motion_process import recover_from_6d,rotation_vector_to_matrix,recover_from_ric
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

    #trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger,inference_mode = False)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
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
    
    log.info("Starting testing!")

    from src.models.metrics.compute import ComputeMetrics,ComputeMetrics_obj,ContactMetric,ContactMetricGT
    
    # trainer.test(model, datamodule=datamodule)[0]
    from src.data.humanml.scripts.motion_process import recover_from_ric
    all_metrics_new={}
    
    # print(output)

    # for key, item in output.items(): 
    #     output[key] = [item.item()]
    # all_metrics_new.update(output)
    # print("--------------------------------------")
    # for key, item in output2.items(): 
    #     output2[key] = [item.item()]
    # print(output2)
    # all_metrics_new.update(output2)
    # print("--------------------------------------")

    all_metrics = {}
    #replication_times = cfg.model.metrics.replicate_times
    replication_times = 10
    output_all={}
    output2_all={}
    output3_all={}
    output3GT_all={}
    for j in range(replication_times):
        metrics = trainer.test(model, datamodule=datamodule)[0]
        log.info(f"Evaluating KPG - Replication {j}") 
        #print(output4)
        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

        
    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval

    print_table(f"Mean Metrics", all_metrics_new)
    
    #all_metrics_new.update(all_metrics)
    # save metrics to file
    metric_file = os.path.join(cfg.paths.output_dir, f"metrics.json")

    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    log.info(f"Testing done, the metrics are saved to {str(metric_file)}")

    return all_metrics_new, object_dict
    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    # metric_dict = trainer.callback_metrics
    # return metric_dict, object_dict


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
