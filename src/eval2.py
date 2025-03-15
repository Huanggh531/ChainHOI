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
import torch.optim as optim
import trimesh
import torch.nn.functional as F
from torchsdf import index_vertices_by_faces, compute_sdf
from rich import get_console
from rich.table import Table
from scipy.spatial.transform import Rotation
from einops import repeat, rearrange
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

def closest_point_on_edge(points, v0, v1, v2):
    # points [B, T, N, M, 3] v0, v1, v2 [B, T, M, 3]
    B, T, N, M = points.shape[:4]
    edges = rearrange([v1 - v0, v2 - v0, v2 - v1], "K B T M D -> B T M K D")
    point_to_edge = repeat(points, "B T N M D -> B T N M K D", K=3) - repeat(rearrange([v0, v1, v1], "K B T M D -> B T M K D"),"B T M K D -> B T N M K D",N=22)
    
    t = torch.einsum("btnmkd,btmkd->btnmk", point_to_edge, edges) / torch.einsum("btmkd,btmkd->btmk", edges, edges).unsqueeze(dim=2)
    t = torch.clamp(t, 0, 1) # b t n m k
    closest_point = t.unsqueeze(dim=-1) * edges.unsqueeze(2) + rearrange([v0, v0, v1], "K B T M D -> B T M K D").unsqueeze(2)
    dist = (points.unsqueeze(dim=4) - closest_point).norm(dim=-1)
    return dist.min(dim=-1)[0]


def point_to_triangle_distance(points, triangles):
    # points [B, T, N, 3] triangles [B, T, M, 3, 3]
    B, T, N, M = points.shape[0], points.shape[1], points.shape[2], triangles.shape[2]
    v0, v1, v2 = triangles[:, :, :, 0], triangles[:, :, :, 1], triangles[:, :, :, 2]
    # 边向量和法线向量
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = torch.cross(edge1, edge2)
    normal = normal / torch.norm(normal, dim=-1, keepdim=True) # B T M 3

    n_points = repeat(points, "b t n d -> b t n m d", m=M)
    n_norms = repeat(normal, "b t m d -> b t n m d", n=N)
    n_v0 = repeat(v0, "b t m d -> b t n m d", n=N)
    
    to_v0 = n_points - n_v0 # B T N M 3
    ori_dist_to_plan = torch.einsum("btnmd,btnmd->btnm", to_v0, n_norms)
    dist_to_plane = ori_dist_to_plan
    
    # 计算投影点是否在三角形内部
    projection = n_points - dist_to_plane[..., None] * n_norms
    
    # 计算重心坐标
    def barycentric_coords(p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = torch.sum(v0 * v0, dim=-1)
        d01 = torch.sum(v0 * v1, dim=-1)
        d11 = torch.sum(v1 * v1, dim=-1)
        d20 = torch.sum(v2 * v0, dim=-1)
        d21 = torch.sum(v2 * v1, dim=-1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        return u, v, w
    
    u, v, w = barycentric_coords(projection, n_v0, v1.unsqueeze(dim=2), v2.unsqueeze(dim=2))
    
    # 判断投影是否在三角形内部，不在内部则置为inf
    inside = (u >= 0) & (v >= 0) & (w >= 0)
    distances_to_edges = torch.where(~inside, torch.tensor(float('inf'), device=points.device), dist_to_plane)
    
    # 计算点到三角面片上的三条边上的最小距离
    dist_1 = closest_point_on_edge(n_points, v0, v1, v2)
    # 加上距离正负号
    dist_1 = dist_1 * torch.sign(ori_dist_to_plan + 1e-9)
    
    # 选取绝对值最小的距离 -> 离三角面片最近
    dist_1 = torch.gather(dist_1, -1, dist_1.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    dist_2 = torch.gather(distances_to_edges, -1, distances_to_edges.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    stack_dist = torch.stack([dist_1, dist_2], dim=-1)
    dist = torch.gather(stack_dist, -1, stack_dist.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    
    return dist

def optimization_motion(pred_gcn_motion,sixD,all_obj_points,faces,contact,contact_pro,length):
    pred_gcn_motion=pred_gcn_motion.float()
    h = pred_gcn_motion[...,:263].cuda()
    o = pred_gcn_motion[...,263:269].cuda()
    h.requires_grad=True
    o.requires_grad=True
 
   # Velocity Loss Function
    def velocity_loss(h, o,length):
        
        batch_size=h.shape[0]
        loss_h=torch.tensor(0.0,requires_grad=True,device=h.device).float()
        loss_o=torch.tensor(0.0,requires_grad=True,device=o.device).float()
        for i in range(batch_size):
            # 获取当前序列的有效长度
            T = length[i]
            # 计算当前序列的 h 和 o 的连续帧差异的 L1 距离
            diff_h = torch.abs(h[i, 1:T] - h[i, :T-1])
            diff_o = torch.abs(o[i, 1:T] - o[i, :T-1])
            # 计算 L1 距离并累加到总损失
            loss_h = loss_h + torch.sum(diff_h)
            loss_o = loss_o + torch.sum(diff_o)
        loss_h = 1.0 * loss_h / batch_size
        loss_o = 1.0 * loss_o / batch_size
        
        # 返回总损失
        return loss_h + loss_o
    def contact_loss(h,o,sixD,all_obj_points,faces,contact,contact_pro,length):
        human_positions=recover_from_6d(h,sixD,22).float()
        angle, trans = o[..., 0:3], o[..., 3:6]
        rot = rotation_vector_to_matrix(angle)
        B=rot.shape[0]
        T=rot.shape[1]
        # 使用 PyTorch 的矩阵乘法    
        dis_loss=torch.tensor(0.0,requires_grad=True,device='cuda')
        pene_distance=torch.tensor(0.0,requires_grad=True,device='cuda')
        obj_point = torch.matmul(all_obj_points.unsqueeze(1).expand(-1, T, -1, -1), rot.transpose(2, 3)) + trans.unsqueeze(2)
        for b in range(obj_point.shape[0]):
            faces_expanded = faces[b].unsqueeze(0).unsqueeze(-1).expand(obj_point.shape[1], -1, -1, 3)  # [T, M, 3, 3]
            # 使用torch.gather获取对应的点
            obj_point_expanded = obj_point[b].unsqueeze(1).expand(-1, faces[b].shape[0], -1, -1)  # [T, M, N, 3]
            indexed_vertices = torch.gather(obj_point_expanded, 2, faces_expanded)
            distances=point_to_triangle_distance(human_positions[b:b+1],indexed_vertices.unsqueeze(0))
            # # 计算所有关节点的 MSE 损失
            #mse_loss = F.l1_loss(torch.zeros_like(distances[0]), distances[0], reduction='none')
            
            mse_loss = F.mse_loss(contact[b], torch.abs(distances[0]), reduction='none')
            contact_mask = (contact_pro[b] > 0).float()
            contact_mask[length[b]:]=0
            masked_loss = mse_loss * contact_mask
            if contact_mask.sum() > 0:
                dis_loss = dis_loss + masked_loss.sum() / contact_mask.sum()
            # 计算小于零的值的平均值

            sdf_loss=distances[0][:length[b]]
            negative_values = sdf_loss[sdf_loss < 0]  # 获取小于零的值
            pene_distance = pene_distance + ((-1 * torch.mean(negative_values)) if negative_values.numel() > 0 else torch.tensor(0.0))  # 计算平均值
        dis_loss = dis_loss / (obj_point.shape[0])
        pene_distance = pene_distance / (obj_point.shape[0])
        return dis_loss,pene_distance
       
    # 创建优化器
    optimizer = optim.AdamW([h, o], lr=0.001)
    lambda_vel=0.01
    lambda_con = 1.0
    lambda_sdf = 1.0
    epoch = 1
    contact_clone=[]
    faces_clone=[]
    for i in range(len(contact)):
        contact_clone.append(contact[i].clone().float())
    for i in range(len(faces)):
        faces_clone.append(faces[i].clone())
    # 优化循环
    while True:  # 假设优化100轮
        optimizer.zero_grad()  # 清空梯度
        con_loss,sdf_loss = contact_loss(h,o,sixD.float(),all_obj_points,faces_clone,contact_clone,contact_pro,length)
        #vel_loss = velocity_loss(h, o,length)
        #loss = lambda_vel * vel_loss + lambda_con * con_loss + lambda_sdf * sdf_loss   # 计算损失
        loss = lambda_con * con_loss + lambda_sdf * sdf_loss   # 计算损失
        #print(loss,vel_loss,con_loss,sdf_loss)
        # if loss < 1.0 or epoch >= 100:
        #     break
        if loss < 0.4 or epoch >= 100:
            break
        epoch += 1

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        #print(h[:,0:10])
        #print(o[:,0:10])
    return torch.cat([h,o],dim=-1)

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
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger,inference_mode=False)
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

    from src.models.metrics.compute import ComputeMetrics,ComputeMetrics_obj,ContactMetric
    from src.models.metrics.tm2t import TM2TMetrics
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
    replication_times = cfg.model.metrics.replicate_times
    #replication_times=2
    output_topk={}
    output_all={}
    output2_all={}
    output3_all={}
    
    
    for j in range(replication_times):
        #t2m_evaluator=MotionCLIPEvaluator(dataset_name,evaluator.T2M_dir)
        trainer.test(model, datamodule=datamodule)[0]
        dataset=trainer.test_dataloaders.dataset
        t2m_evaluator=model.t2m_evaluator
        t2m_metrics = TM2TMetrics(diversity_times=100,dist_sync_on_step=True).cuda()
        #metric_compute=ComputeMetrics(njoints=22,jointstype="humanml3d").cuda()
        #metric_compute2=ComputeMetrics_obj(njoints=22,jointstype="humanml3d").cuda()
        metric_compute3=ContactMetric(dist_sync_on_step=True).cuda()
        for i in range(len(model.gt_motion)):
            print("---------------{} / {}---------------".format(i,len(model.gt_motion)))
            pred_motion_all=model.pred_motion[i].clone()
            
            all_obj_points=model.all_obj_points[i]
            faces=model.faces[i]
            sixD=model.sixD[i].clone()
            contact=model.contact[i]
            contact_pro=model.contact_pro[i]
            lengths=model.length[i]
            text=model.text[i]
            gt_motion_all=model.gt_motion[i]
            #pred_motion_all = optimization_motion(pred_motion_all,sixD,all_obj_points,faces,contact,contact_pro,lengths)
            pred_motion_all = optimization_motion(pred_motion_all,sixD,all_obj_points,faces,contact,model.pred_contact[i],lengths)

            pred_motion_all = pred_motion_all.cpu().detach().cuda()
            t2m_text_emb = t2m_evaluator.extract_text_embedding(pred_motion_all.device,text)
            t2m_motion_gen_emb = t2m_evaluator.extract_motion_embedding(dataset.norm_transform_th(pred_motion_all[...,0:269])[...,0:263], lengths)
            t2m_motion_gt_emb = t2m_evaluator.extract_motion_embedding(dataset.norm_transform_th(gt_motion_all[...,0:269].cuda())[...,0:263], lengths)
            t2m_metrics.update(t2m_text_emb, t2m_motion_gen_emb, t2m_motion_gt_emb, lengths)
            
            pred_motion=pred_motion_all[...,0:263]
            pred_motion=recover_from_ric(pred_motion.float().cpu().detach(),22).cuda()
            
            gt_motion=gt_motion_all[...,0:263]
            gt_motion=recover_from_ric(gt_motion.float(),22)
            pred_obj=pred_motion_all[...,263:].cuda()
            gt_obj=gt_motion_all[...,263:]
            
            obj_points=model.obj_points[i]
            obj_name=model.obj_name[i]
            #metric_compute.update(pred_motion,gt_motion.cuda(),lengths)
            
            
            
            #metric_compute2.update(pred_obj,gt_obj.cuda(),obj_points.cuda(),obj_name,lengths)
            human_positions=pred_motion
            angle, trans = pred_obj[..., 0:3].float(), pred_obj[..., 3:6].float()
            rot = rotation_vector_to_matrix(angle)
            B=rot.shape[0]
            T=rot.shape[1]
            # 使用 PyTorch 的矩阵乘法
            obj_point = torch.matmul(all_obj_points.unsqueeze(1).expand(-1, T, -1, -1), rot.transpose(2, 3)) + trans.unsqueeze(2)
            metric_compute3.update(model.pred_contact[i],contact_pro,lengths,human_positions,obj_point,faces)
            
        
        metric_output = t2m_metrics.compute(sanity_flag=False) 
        #output=metric_compute.compute(sanity_flag=False)
        #output2=metric_compute2.compute(sanity_flag=False)
        output3=metric_compute3.compute(sanity_flag=False)
        
        model.pred_motion.clear()
        model.gt_motion.clear()
        model.text.clear()
        model.length.clear()
        model.obj_points.clear()
        model.obj_name.clear()
        model.pred_contact.clear()
        model.contact_pro.clear()
        model.faces.clear()
        model.all_obj_points.clear()
        model.sixD.clear()
        model.contact.clear()

        # 清理显存
        torch.cuda.empty_cache()
        # model.pred_motion=[]
        # model.gt_motion=[]
        # model.text=[]
        # model.length=[]
        # model.obj_points=[]
        # model.obj_name=[]
        # model.pred_contact=[]
        # model.contact_pro=[]
        # model.faces=[]
        # model.all_obj_points=[]
        # model.sixD=[]
        # model.contact=[]

        for key, item in metric_output.items():
            if key not in output_topk:
                if torch.is_tensor(item):
                    output_topk[key] = [item.cpu()]
                else:
                    output_topk[key] = [torch.tensor(item)]
            else:
                if torch.is_tensor(item):
                    output_topk[key] += [item.cpu()]
                else:
                    output_topk[key] += [torch.tensor(item)]
        # for key, item in output.items():
        #     if key not in output_all:
        #         output_all[key] = [item.cpu()]
        #     else:
        #         output_all[key] += [item.cpu()]
        # for key, item in output2.items():
        #     if key not in output2_all:
        #         output2_all[key] = [item.cpu()]
        #     else:
        #         output2_all[key] += [item.cpu()]
        for key, item in output3.items():
            if key not in output3_all:
                output3_all[key] = [item.cpu()]
            else:
                output3_all[key] += [item.cpu()]
        log.info(f"--------Evaluating KPG - Replication {j}---------") 
        print(output_topk)
        print(output3_all)
        
        


    for key, item in output_topk.items():
        mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
        all_metrics_new[key + "/mean"] = np.float64(mean)
        all_metrics_new[key + "/conf_interval"] =  np.float64(conf_interval)

    # for key, item in output_all.items():
    #     mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
    #     all_metrics_new[key + "/mean"] = np.float64(mean)
    #     all_metrics_new[key + "/conf_interval"] =  np.float64(conf_interval)

    # for key, item in output2_all.items():
    #     mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
    #     all_metrics_new[key + "/mean"] =  np.float64(mean)
    #     all_metrics_new[key + "/conf_interval"] =  np.float64(conf_interval)
    
    for key, item in output3_all.items():
        mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
        all_metrics_new[key + "/mean"] =  np.float64(mean)
        all_metrics_new[key + "/conf_interval"] =  np.float64(conf_interval)

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
