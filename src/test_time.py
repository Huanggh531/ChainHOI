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

import tqdm
@torch.no_grad()
def calculate_run_time(_run, repetitions):
    torch.backends.cudnn.benchmark = True

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in tqdm.tqdm(range(2)):
            _run(run_id=0)
            # _ = model(model_input[0],model_input[1])

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            # _ = model(model_input[0],model_input[1])
            _run(run_id=rep)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time
    avg = timings.sum() / repetitions
    print('\navg={}ms\n'.format(avg))


from functools import partial
def pre_process(dataloader,model,guidance,device):
    motion_info = []

    for i, data in enumerate(dataloader):
        # motion, text, length, word_embs, pos_ohot, text_len, tokens = data
        info={}
        info["gt_gcn_motion"]=data["gcn_motion"].to(device)
        info["length"]=data["motion_len"].to(device)
        info["text"]=data["text"]
        # with torch.no_grad():
        #     obj_emb = model.objEmbedding(data['obj_points'].to(device))
        info["obj_points"] = data['obj_points'].to(device)
        info["all_obj_points"] = data["all_obj_points"].to(device)
        info["obj_name"] = data["seq_name"]
        obj_name = data["seq_name"]

        if guidance:
            afford_sample = []
            for b in range(len(obj_name)):
                afford = np.load(os.path.join("dataset/behave_t2m/guidance",obj_name[b]+".npy"))
                afford = torch.tensor(afford,dtype=info["gt_gcn_motion"].dtype).to(info["gt_gcn_motion"].device)
                afford_sample.append(afford)
            afford_sample = torch.stack(afford_sample)
            info["afford_sample"] = afford_sample
        motion_info.append(info)
    print(len(motion_info))
    return motion_info


def run_net(model, motion_info,dataset,guidance, run_id):
    for i in range(len(motion_info)):
        gt_gcn_motion = motion_info["gt_gcn_motion"]
        obj_emb = motion_info["obj_points"]
        text = motion_info["text"]
        length = motion_info["length"]
        obj_name = motion_info["obj_name"]
        all_obj_points = motion_info["all_obj_points"]
        if guidance:
            afford_sample = motion_info["afford_sample"]
            model.sample_motion(gt_gcn_motion, length, text,obj_emb,all_obj_points,None,None,dataset,None,obj_name,afford_sample)
        else:
            model.sample_motion(gt_gcn_motion, length, text,obj_emb,all_obj_points,None,None,dataset,None,obj_name)

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

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger,inference_mode = False)
    
    log.info("Starting testing!")

    datamodule.setup(None)
    datamodule.hparams.test_batch_size = 1
    model.to(f"cuda:{trainer.device_ids[0]}")
    model.eval()
    dataloader = datamodule.test_dataloader2()

    #guidance:with/wo AIC
    guidance=True #True:with AIC


    motion_info = pre_process(dataloader,model,guidance,model.device)
    motion_info = motion_info[0]
    run_func = partial(run_net, model=model, motion_info=motion_info,dataset=datamodule.test_dataset,guidance=guidance)
    calculate_run_time(run_func, 10)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_time.yaml")
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
