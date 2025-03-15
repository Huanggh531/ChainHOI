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
import trimesh
from scipy.spatial.transform import Rotation
from rich import get_console
from rich.table import Table
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def text_to_object(text):
    obj_list = ['backpack','basketball','boxlarge','boxtiny','boxlong','boxsmall','boxmedium','chairblack','chairwood',
        'monitor','keyboard','plasticcontainer','stool','tablesquare','toolbox','suitcase','tablesmall','yogamat','yogaball','trashbin', 'clothesstand', 'floorlamp', 'tripod']
    
    all_obj_points = []
    all_obj_normals = []
    all_obj_names = []
    import re

    for i in range(len(text)):

        for j in range(len(obj_list)):
            if obj_list[j] in text[i]:
                name = obj_list[j]
                break
        
        # load obj points----------------
        obj_path = '/home/guohong/HOI/dataset/behave_t2m/object_mesh'
        obj_name = name
        mesh_path = os.path.join(obj_path, simplified_mesh[obj_name])
        import trimesh
        temp_simp = trimesh.load(mesh_path)
        obj_points = np.array(temp_simp.vertices).astype(np.float32) * 0.15
        obj_faces = np.array(temp_simp.faces).astype(np.float32)
        obj_normals = obj_faces


        # sample object points
        obj_sample_path = '/home/guohong/HOI/dataset/behave_t2m/object_sample/{}.npy'.format(name)
        choose = np.load(obj_sample_path)
        

        # center the meshes
        center = np.mean(obj_points, 0)
        obj_points -= center

                
        obj_points = obj_points[choose] 
        obj_normals = obj_normals[choose] 


        all_obj_points.append(obj_points)
        all_obj_normals.append(obj_normals)
        all_obj_names.append(obj_name)

    return np.array(all_obj_points),  np.array(all_obj_normals),  np.array(all_obj_names)

simplified_mesh = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
    'clothesstand':"clothesstand_cleaned_simplified.obj",
    'floorlamp':"floorlamp_cleaned_simplified.obj",
    'tripod':"tripod_cleaned_simplified.obj",
    'whitechair':"whitechair_cleaned_simplified.obj",
    'woodchair':"woodchair_cleaned_simplified.obj"
}

def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q, order, epsilon=0, deg=True):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    if deg:
        return torch.stack((x, y, z), dim=1).view(original_shape) * 180 / np.pi
    else:
        return torch.stack((x, y, z), dim=1).view(original_shape)

def recover_root_rot_pos(data):

    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    
    return positions



def plot_3d_motion(save_path, kinematic_tree, joints, obj_points, hc_mask, oc_mask, title,  figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    # print(f" motion :{joints.shape} obj :{obj_points.shape}: hc {hc_mask.shape} oc: {oc_mask.shape}")

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)

    data = joints.copy().reshape(len(joints), -1, 3)

    # data *= 1.3  # scale for visualization

    
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    # trajec = data[:, 0, [0, 2]]

    # if hint is not None:
    #     hint[..., 0] -= data[:, 0, 0]
    #     hint[..., 2] -= data[:, 0, 2]

    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5


        plot_xzPlane(MINS[0], MAXS[0] , 0, MINS[2],
                     MAXS[2] )




        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)
        if obj_points is not None:

            x2 = obj_points[ index, :, 0]
            y2 = obj_points[ index, :, 1]
            z2 = obj_points[ index, :, 2]
            ax.scatter(x2, y2, z2, color='grey', s=1, alpha=0.5)


            

        if hc_mask is not None:
            # hc_idx = np.where(hc_mask == 1.0)
            # if hc_idx != None:
            x3 = data[index, hc_mask, 0]
            y3 = data[index, hc_mask, 1]
            z3 = data[index, hc_mask, 2]
            ax.scatter(x3, y3, z3, color='red', s=6 ,alpha=1.0)
        if oc_mask is not None and oc_mask[0]!=-1:
            x4 = x2[oc_mask]
            y4 = y2[oc_mask]
            z4 = z2[oc_mask]
            ax.scatter(x4, y4, z4, color='blue', s=6, alpha=1.0)


        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)

    plt.close()

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
    model.load_state_dict(state_dict,strict=False)

    log.info("Starting visualize!")
    all_metrics = {}
    


    
    
    #log.info(f"Evaluating MultiModality - Replication {i}")
    #datamodule.mm_mode(True, cfg.model.metrics.mm_num_samples)
    #mm_metrics = trainer.test(model, datamodule=datamodule)[0]
    #metrics.update(mm_metrics)
    #datamodule.mm_mode(False)
    


    trainer.test(model, datamodule=datamodule)
    
    gt_motion=[]
    pred_motion=[]
    text=[]
    lengths=[]
    obj_name=[]
    obj_points=[]

    for j in range(len(model.gt_motion)):
        for i in range(len(model.gt_motion[j])):
            gt_motion.append(model.gt_motion[j][i])
            pred_motion.append(model.pred_motion[j][i])
            text.append(model.text[j][i])
            lengths.append(model.length[j][i])
            obj_points.append(model.obj_points[j][i])
            obj_name.append(model.obj_name[j][i])
        break

    gt_motion=torch.stack(gt_motion).unsqueeze(1).float()
    pred_motion=torch.stack(pred_motion).unsqueeze(1).float()
    # obj_points, obj_normals, obj_name = text_to_object(text)

    
    sample_obj=pred_motion[...,263:]
    sample_obj = sample_obj
    sample = pred_motion[..., :263]
    n_joints = 22
    import pdb
    pdb.set_trace()
    sample = recover_from_ric(sample, n_joints)
    import pdb
    pdb.set_trace()
    sample = sample[:,:,:,:n_joints*3]
    sample = sample.reshape(sample.shape[0], sample.shape[1], sample.shape[2], n_joints, 3)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

    
    gt_sample_obj=gt_motion[...,263:]
    gt_sample_obj = gt_sample_obj
    gt_sample = gt_motion[..., :263]
    n_joints = 22

    gt_sample = recover_from_ric(gt_sample, n_joints)
    gt_sample = gt_sample[:,:,:,:n_joints*3]
    gt_sample = gt_sample.reshape(gt_sample.shape[0], gt_sample.shape[1], gt_sample.shape[2], n_joints, 3)
    gt_sample = gt_sample.view(-1, *gt_sample.shape[2:]).permute(0, 2, 3, 1)
    


    skeleton = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    for i in range(32):

        caption = text[i]
        length = lengths[i]
        motion = sample[i].numpy().transpose(2, 0, 1)[:length]#(196,22,3)
        motion_obj = (sample_obj[i].squeeze(0).numpy())[:length,:]  
        vertices = obj_points[i].cpu().numpy()#(512,3)
        mesh_path = os.path.join("./dataset/behave_t2m/object_mesh", simplified_mesh[obj_name[i].split('_')[2]])
        
        temp_simp = trimesh.load(mesh_path)
        all_vertices = temp_simp.vertices
        # center the meshes
        center = np.mean(all_vertices, 0)
        all_vertices -= center
        new_vertices = np.concatenate([all_vertices, vertices[-2:]], 0)



        # transform
        import pdb
        pdb.set_trace()
        angle, trans = motion_obj[:, :3].transpose(1,0), motion_obj[:, 3:].transpose(1,0)
        rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
        obj_point = np.matmul(vertices[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]



        save_file = str(i)+"_"+str(obj_name[i].split('_')[2])+".mp4"
        
        animation_save_path = os.path.join("videos8-obj", save_file)


        plot_3d_motion(animation_save_path, skeleton, motion, obj_point, hc_mask=None, oc_mask=None, title=caption, fps=20)
    print("-------------------------------------")
    # for i in range(32):
    #     caption = text[i]
    #     length = lengths[i]
    #     motion = gt_sample[i].numpy().transpose(2, 0, 1)[:length]#(196,22,3)
    #     vertices = obj_points[i].cpu().numpy()#(512,3)
    #     motion_obj = (gt_sample_obj[i].squeeze(0).numpy())[:length,:]  
    #     mesh_path = os.path.join("./dataset/behave_t2m/object_mesh", simplified_mesh[obj_name[i].split('_')[2]])
    #     temp_simp = trimesh.load(mesh_path)
    #     all_vertices = temp_simp.vertices
    #     # center the meshes
    #     center = np.mean(all_vertices, 0)
    #     all_vertices -= center
    #     new_vertices = np.concatenate([all_vertices, vertices[-2:]], 0)



    #     # transform
    #     angle, trans = motion_obj[:, :3].transpose(1,0), motion_obj[:, 3:].transpose(1,0)
    #     rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
    #     obj_point = np.matmul(vertices[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]



    #     save_file = str(i)+"_"+str(obj_name[i].split('_')[2])+".mp4"
        
    #     animation_save_path = os.path.join("gt_videos10", save_file)


    #     plot_3d_motion(animation_save_path, skeleton, motion, obj_point, hc_mask=None, oc_mask=None, title=caption, fps=20)


    


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
