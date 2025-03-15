'''
在gcn5的基础上，在sample的后十步进行优化
'''



import os

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import MeanMetric, MetricCollection, MinMetric
import lightning.pytorch as L
from .nets.points_encoder import PointNet2Encoder
from .metrics import TM2TMetrics, MMMetrics,ComputeMetrics,ComputeMetrics_obj,ContactMetric,PeneMetric

from .utils.utils import occumpy_mem, CosineWarmupScheduler, replace_annotation_with_null, lengths_to_mask
from src.data.humanml.scripts.motion_process import recover_from_6d,rotation_vector_to_matrix,recover_from_ric
import trimesh
from scipy.spatial.transform import Rotation
from torchsdf import index_vertices_by_faces, compute_sdf
from einops import repeat, rearrange
import torch.optim as optim
import torch.nn as nn
#from .nets.pct import Pct
#from .nets.points_encoder import PointNet2Encoder
from .nets.points_encoder import PointNet2SemSegMSG
from .condition import Guide_Contact
@torch.no_grad()
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
    _n = n_points.shape[1] // 2
    # dist_1_a = closest_point_on_edge(n_points[:, :_n], v0, v1, v2)
    # dist_1_b = closest_point_on_edge(n_points[:, _n:], v0, v1, v2)
    # dist_1 = torch.cat([dist_1_a, dist_1_b], dim=1)
    
    # 加上距离正负号
    dist_1 = dist_1 * torch.sign(ori_dist_to_plan + 1e-9)
    
    # 选取绝对值最小的距离 -> 离三角面片最近
    dist_1 = torch.gather(dist_1, -1, dist_1.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    dist_2 = torch.gather(distances_to_edges, -1, distances_to_edges.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    stack_dist = torch.stack([dist_1, dist_2], dim=-1)
    dist = torch.gather(stack_dist, -1, stack_dist.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    
    return dist

class CHAINHOI(L.LightningModule):
    def __init__(self,
                 text_encoder,
                 gcn_denoiser,
                 noise_scheduler,
                 sample_scheduler,
                 #Pct,#for pointnet
                 text_replace_prob,
                 guidance_scale,
                 dataset_name, # for evaluate model
                 evaluator,
                 optimizer, # for optimize model
                 latent_dim,
                 lr_scheduler=None,
                 obj_flag=False,
                 debug=False,
                 generate=False,
                 dis_alpha=2.0,
                 obj_alpha=1.0,
                 ddimstep=50,
                 **kwargs):
        super(CHAINHOI, self).__init__()
        self.save_hyperparameters(logger=False, ignore=['text_encoder', 'gcn_denoiser'])
        self.text_encoder = text_encoder
        self.gcn_denoiser = gcn_denoiser

        self.noise_scheduler = noise_scheduler
        if sample_scheduler is False:
            self.sample_scheduler = noise_scheduler
        else:
            self.sample_scheduler = sample_scheduler
        self.sample_scheduler.set_timesteps(ddimstep)

        self.configure_evaluator_and_metrics(dataset_name, evaluator)
        self.dis_alpha = dis_alpha
        self.obj_alpha = obj_alpha
        self.is_mm_metric = False # only used during mm testing
        self.pred_motion=[]
        self.gt_motion=[]
        self.text=[]
        self.length=[]
        self.obj_points=[]
        self.obj_name=[]
        self.all_obj_points=[]
        self.faces=[]
        self.contact_pro=[]
        self.pred_contact=[]
        self.sixD=[]
        self.contact=[]
        self.t2m_model=None
        #for pointnet
        self.objEmbedding = PointNet2SemSegMSG(hparams={"model.use_xyz":True})
        #self.objEmbedding = Pct
        self.obj_flag=obj_flag


    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            self.gcn_denoiser = torch.compile(self.gcn_denoiser)
            self.objEmbedding = torch.compile(self.objEmbedding)

    def configure_optimizers(self):
        params = list(self.objEmbedding.parameters()) + list(self.gcn_denoiser.parameters())
        optimizer = self.hparams.optimizer(params)
        if self.hparams.lr_scheduler is not None:
            lr_scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return optimizer

    def configure_evaluator_and_metrics(self, dataset_name, evaluator):
        self.train_metrics = MetricCollection({"loss": MeanMetric()})

        from src.models.evaluator.MotionCLIP import MotionCLIPEvaluator
        self.t2m_evaluator=MotionCLIPEvaluator(dataset_name,evaluator.T2M_dir)
        self.t2m_metrics = TM2TMetrics(diversity_times=300 if dataset_name == "hml3d" else 100,
                                dist_sync_on_step=True)
        self.concat_metric=ContactMetric(dist_sync_on_step=True)
        self.pene_metric=PeneMetric(dist_sync_on_step=True)
        

    def on_train_start(self):
        if self.hparams.ocpm:
            occumpy_mem(self.device.index)

    def calculate_diffusion_loss(self, output, noise, gt_motion, padding_mask, weight=1):
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            difuss_loss = F.mse_loss(noise, output,reduction="none")
        elif prediction_type == "sample":
            difuss_loss = F.mse_loss(gt_motion, output,reduction="none")
        else:
            raise ValueError(f"{prediction_type} not supported!")

        difuss_loss = difuss_loss.mean(dim=(2,3))
        loss = difuss_loss[padding_mask].sum() / padding_mask.sum()
        return weight * loss    

    def calculate_dis_loss(self, output, length, all_obj_points, motion_unorm_gt, faces, contact, contact_pro,padding_mask, weight1=1,weight2=1):
        angle_gt, trans_gt = motion_unorm_gt[..., 263:266], motion_unorm_gt[..., 266:269]
        rot_gt = rotation_vector_to_matrix(angle_gt)
        B, T =rot_gt.shape[:2] 
        # 使用 PyTorch 的矩阵乘法
        obj_point_gt = torch.matmul(all_obj_points.unsqueeze(1).expand(-1, T, -1, -1), rot_gt.transpose(2, 3)) + trans_gt.unsqueeze(2)
        
        dataset = self.trainer.train_dataloader.dataset
        pred_motion = dataset.to_hml3d_format(output)
        motion_unorm = dataset.inv_transform_th(pred_motion[...,0:269]).float()

        angle, trans = motion_unorm[..., 263:266], motion_unorm[..., 266:269]
        rot = rotation_vector_to_matrix(angle)
        obj_point = torch.matmul(all_obj_points.unsqueeze(1).expand(-1, T, -1, -1), rot.transpose(2, 3)) + trans.unsqueeze(2)
        sixD_pred = pred_motion[..., 269:273]

        human_positions = recover_from_6d(motion_unorm,sixD_pred,22).float() 
        #obj_loss = F.mse_loss(obj_point, obj_point_gt,reduction='none')   
        obj_loss = F.mse_loss(obj_point, obj_point_gt,reduction='none')

        mask = padding_mask.unsqueeze(-1).unsqueeze(-1)  
        obj_loss = obj_loss.mean(dim=(2,3))
        obj_loss = obj_loss * padding_mask  
        obj_loss = obj_loss.mean()

        dis_loss = 0.0
        for b in range(obj_point_gt.shape[0]):
            faces_expanded = faces[b].unsqueeze(0).unsqueeze(-1).expand(obj_point_gt.shape[1], -1, -1, 3)  # [T, M, 3, 3]
            # 使用torch.gather获取对应的点
            obj_point_expanded = obj_point_gt[b].unsqueeze(1).expand(-1, faces[b].shape[0], -1, -1)  # [T, M, N, 3]
            indexed_vertices = torch.gather(obj_point_expanded, 2, faces_expanded)
            distances = point_to_triangle_distance(human_positions[b:b+1], indexed_vertices.unsqueeze(0))
            mse_loss = F.mse_loss(torch.zeros_like(distances[0]), torch.abs(distances[0]), reduction='none') 
           
            # 创建掩码矩阵
            contact_mask = (contact_pro[b] > 0).float()
            contact_mask[length[b]:] = 0
            masked_loss = mse_loss * contact_mask 
            if contact_mask.sum() > 0:
                dis_loss = dis_loss + masked_loss.sum() / contact_mask.sum()
        dis_loss = dis_loss / (obj_point.shape[0])
        
        return dis_loss * weight1,obj_loss * weight2

    def _step_network(self, batch, batch_idx):
        gcn_motion, length, text,all_obj_points,contact_pro = batch["gcn_motion"], batch["motion_len"], batch["text"],batch["all_obj_points"],batch["contact_pro"]
        obj_points=batch["obj_points"]
        faces=batch["faces"]
        motion_unorm_gt=batch["motion_unorm_gt"]
        contact=batch["contact"]

        #按self.hparams.text_replace_prob概率将条件变为null：classifer-free
        text = replace_annotation_with_null(text, self.hparams.text_replace_prob)
        obj_emb = None

        if self.obj_flag:
            obj_emb = self.objEmbedding(obj_points) # [bs, n_points, d]
        with torch.no_grad():
            text_embed = self.text_encoder(text, self.device)

        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,(gcn_motion.size(0),), device=gcn_motion.device).long()
        padding_mask = lengths_to_mask(length, self.device)
        noise = torch.randn_like(gcn_motion, device=gcn_motion.device) 

        x_t = self.noise_scheduler.add_noise(gcn_motion, noise, timestep)

        output = self.gcn_denoiser(x_t, padding_mask, timestep, text_embed,obj_emb)
        
        diffusion_loss = self.calculate_diffusion_loss(output, noise, gcn_motion, padding_mask, weight=1.0)
        dis_loss,obj_loss = self.calculate_dis_loss(output, length, all_obj_points, motion_unorm_gt, faces,
                                          contact, contact_pro,padding_mask, weight1=self.dis_alpha,weight2=self.obj_alpha)

        return {"loss": diffusion_loss+dis_loss+obj_loss,"difuss_loss":diffusion_loss,"dis_loss":dis_loss,"obj_loss":obj_loss}

    def training_step(self, batch, batch_idx):
        
        losses = self._step_network(batch, batch_idx)
        self.log(f"train/loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"train/difuss_loss", losses["difuss_loss"], prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"train/dis_loss", losses["dis_loss"], prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"train/obj_loss", losses["obj_loss"], prog_bar=True, on_step=True, on_epoch=False)
        
       
        return losses["loss"]

    def on_train_epoch_end(self):
        if self.current_epoch > 0 and self.current_epoch % self.hparams.save_every_n_epochs == 0:
            self.trainer.save_checkpoint(os.path.join(self.hparams.ckpt_path, f"epoch-{self.current_epoch}.ckpt"))

    def on_validation_start(self) -> None:
        if self.hparams.debug:
            self.trainer.save_checkpoint(os.path.join(self.hparams.ckpt_path, f"epoch-{self.current_epoch}.ckpt"))

    def validation_step(self, batch, batch_idx):
        
        self.evaluate(batch, split='val')

    def on_validation_epoch_end(self):
        
        results = {}
        metric_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
        metric_output4=self.concat_metric.compute(sanity_flag=self.trainer.sanity_checking)
        #metric_output5=self.pene_metric.compute(sanity_flag=self.trainer.sanity_checking)
        results.update({f"Metrics/{key}": value.item() for key, value in metric_output.items()})
        results.update({f"Metrics/{key}": value.item() for key, value in metric_output4.items()})
        #results.update({f"Metrics/{key}": value.item() for key, value in metric_output5.items()})
        self.t2m_metrics.reset()
        self.concat_metric.reset()
        #self.pene_metric.reset()
        results.update({"epoch": self.trainer.current_epoch, "step": self.global_step,})

        if self.trainer.sanity_checking is False:
            self.log_dict(results, sync_dist=True)

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, split="test") 

    def on_test_epoch_end(self):  

        results = {}
        if self.is_mm_metric:
            results.update(self.t2m_mm_metric.compute(sanity_flag=self.trainer.sanity_checking)["MultiModality"])
            self.t2m_mm_metric.reset()
        else:
            metric_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
            metric_output4=self.concat_metric.compute(sanity_flag=self.trainer.sanity_checking)
            #metric_output5=self.pene_metric.compute(sanity_flag=self.trainer.sanity_checking)
            results.update({f"Metrics/{key}": value.item() for key, value in metric_output.items()})
            results.update({f"Metrics/{key}": value.item() for key, value in metric_output4.items()})
            #results.update({f"Metrics/{key}": value.item() for key, value in metric_output5.items()})
            self.t2m_metrics.reset()
            self.concat_metric.reset()
            #self.pene_metric.reset()
        if self.trainer.sanity_checking is False:
            self.log_dict(results, sync_dist=True, rank_zero_only=True)         

    def evaluate(self, batch, split='val'):
        
        if self.trainer.sanity_checking:
            return {}
        gcn_motion, length, text,contact,all_obj_points,faces = batch["gcn_motion"], batch["motion_len"], batch["text"], batch["contact"],batch["all_obj_points"],batch["faces"]
        obj_points=batch['obj_points']
        contact_pro=batch["contact_pro"]
        obj_name=batch["seq_name"]

       
        if split == 'test':
            dataset = self.trainer.test_dataloaders.dataset
            k=50
            
        else:
            dataset = self.trainer.val_dataloaders.dataset
            k=5
        
        pred_gcn_motion = self.sample_motion(gcn_motion, length, text,obj_points,all_obj_points,faces,contact,dataset,k,obj_name)
        gt_motion = dataset.to_hml3d_format(gcn_motion)
        pred_motion = dataset.to_hml3d_format(pred_gcn_motion)

        
        
        
        gt_motion_unorm=dataset.inv_transform_th(gt_motion[...,0:269])
        
        motion_unorm=dataset.inv_transform_th(pred_motion[...,0:269])
        
        t2m_text_emb = self.t2m_evaluator.extract_text_embedding(self.device,text)
        t2m_motion_gen_emb = self.t2m_evaluator.extract_motion_embedding(pred_motion[...,0:263], length)
        t2m_motion_gt_emb = self.t2m_evaluator.extract_motion_embedding(gt_motion[...,0:263], length)
        self.t2m_metrics.update(t2m_text_emb, t2m_motion_gen_emb, t2m_motion_gt_emb, length)

        #计算human && obj
        pred_motion_all=motion_unorm
        pred_motion_human=pred_motion_all[...,0:263]
        pred_motion_human=recover_from_ric(pred_motion_human.float().cpu(),22).to(self.device)
        gt_motion_all=gt_motion_unorm
        gt_motion_human=gt_motion_all[...,0:263]
        gt_motion_human=recover_from_ric(gt_motion_human.float().cpu(),22).to(self.device)
        pred_obj=pred_motion_all[...,263:]
        gt_obj=gt_motion_all[...,263:]
        obj_points=batch["obj_points"]

        human_positions=pred_motion_human.float()
        angle, trans = pred_obj[..., 0:3].float(), pred_obj[..., 3:6].float()
        rot = rotation_vector_to_matrix(angle)
        B=rot.shape[0]
        T=rot.shape[1]
        # 使用 PyTorch 的矩阵乘法
        all_obj_points=batch["all_obj_points"]
        obj_point = torch.matmul(all_obj_points.unsqueeze(1).expand(-1, T, -1, -1), rot.transpose(2, 3)) + trans.unsqueeze(2)

       

        self.concat_metric.update(None,contact_pro,length,human_positions,obj_point,faces,obj_name,dataset,batch["idx"])  
        #self.pene_metric.update(None,contact_pro,length,human_positions,obj_point,faces)
       
    def sample_motion(self, gt_gcn_motion, length, text,obj_points,all_obj_points,faces,contact,dataset,k,obj_name,afford_sample=None):
        # import pdb
        # pdb.set_trace()

        with torch.no_grad():
            obj_emb = self.objEmbedding(obj_points)
        B, L, J, D = gt_gcn_motion.shape
        repeated_text = text.copy()
        repeated_text.extend([""] * B)

        text_embed = self.text_encoder(repeated_text, self.device)
        time_steps = self.sample_scheduler.timesteps.to(self.device)
        pred_gcn_motion = torch.randn_like(gt_gcn_motion, device=self.device)
        padding_mask = lengths_to_mask(length, self.device)

        prediction_type = self.noise_scheduler.config.prediction_type
        
        #gt_motion = dataset.to_hml3d_format(gt_gcn_motion)
        #gt_motion_unorm=dataset.inv_transform_th(gt_motion[...,0:269])
        

        for i, t in enumerate(time_steps):

            if obj_emb is None:
                output = self.gcn_denoiser(pred_gcn_motion.repeat([2, 1 ,1, 1]), padding_mask.repeat([2, 1,]),t.repeat([2 * B]), text_embed,None)
            else:
                output = self.gcn_denoiser(pred_gcn_motion.repeat([2, 1 ,1, 1]), padding_mask.repeat([2, 1,]),t.repeat([2 * B]), text_embed,obj_emb.repeat(2,1,1))
            if prediction_type == "epsilon":
                cond_eps, uncond_eps = output.chunk(2)
            elif prediction_type == "sample":
                cond_x0, uncond_x0 = output.chunk(2)
                #由于输出的是x0而不是噪声所以需要obtain_eps_when_predicting_x_0获取噪声
                cond_eps, uncond_eps = self.obtain_eps_when_predicting_x_0(cond_x0, uncond_x0, t, pred_gcn_motion)
            else:
                raise ValueError(f"{prediction_type} not supported!")
            pred_noise = uncond_eps + self.hparams.guidance_scale * (cond_eps - uncond_eps)
            pred_gcn_motion = self.sample_scheduler.step(pred_noise, t, pred_gcn_motion).prev_sample.float() 
            # afford_sample = []
            # for b in range(pred_gcn_motion.shape[0]):
            #     afford = np.load(os.path.join("/home/guohong/HOI-Diff/behave/guidance4",obj_name[b]+".npy"))
            #     afford = torch.tensor(afford,dtype=pred_gcn_motion.dtype).to(pred_gcn_motion.device)
            #     afford_sample.append(afford)
            # afford_sample = torch.stack(afford_sample)
            # afford_sample = afford_sample
            # guide_contact =  Guide_Contact(inv_transform_th=dataset.inv_transform_th,
            #                                 mean=dataset.mean,
            #                                 std=dataset.std,
            #                                 classifiler_scale=100,
            #                                 use_global=False,
            #                                 batch_size=pred_gcn_motion.shape[0],
            #                                 afford_sample = afford_sample)
            # pred_motion = dataset.to_hml3d_format(pred_gcn_motion)
            # with torch.set_grad_enabled(True):
            #     if i < len(time_steps)-1:
            #         pred_motion = rearrange(pred_motion,"b t d -> b d t").unsqueeze(2)
            #         n_guide_steps = 1
            #         for _ in range(n_guide_steps):
            #             loss, grad, loss_list = guide_contact(pred_motion[:,:269],
            #                             t,all_obj_points)
            #             min_variance = 0.01
            #             grad = min_variance * grad
            #             tao1 = 1.0
            #             tao2 = 100.0
            #             # NOTE hard code
            #             pred_motion[:,:263] = pred_motion[:,:263] - tao1 * grad[:,:263] 
            #             pred_motion[:,263:269] = pred_motion[:,263:269] - tao2 * grad[:,263:] 
            #     else:
            #         pred_motion = rearrange(pred_motion,"b t d -> b d t").unsqueeze(2)
            #         n_guide_steps = 100
            #         for _ in range(n_guide_steps):
            #             loss, grad, loss_list = guide_contact(pred_motion[:,:269],t,
            #                             all_obj_points)
            #             min_variance = 0.01
            #             grad = min_variance * grad
            #             tao1 = 1.0
            #             tao2 = 100.0
            #             # NOTE hard code
            #             pred_motion[:,:263] = pred_motion[:,:263] - tao1 * grad[:,:263] 
            #             pred_motion[:,263:269] = pred_motion[:,263:269] - tao2 * grad[:,263:] 
            # pred_motion = rearrange(pred_motion[:,:,0],"b d t -> b t d")
            # pred_gcn_motion = dataset.to_gcn_format(pred_motion).float()

        # for b in range(weight_x0.shape[0]):
        #     save_path=os.path.join("gen_weight_train",obj_name[b])
        #     os.makedirs(save_path, exist_ok=True)
        #     np.save(os.path.join(save_path,"weight.npy"),weight_x0[b][:length[b]].cpu().numpy())
        #     np.save(os.path.join(save_path,"contact.npy"),contact[b][:length[b]].cpu().numpy())
        #     with open(os.path.join(save_path,"text.txt"), "w") as file:
        #         file.write(text[b])

        return pred_gcn_motion

    def obtain_eps_when_predicting_x_0(self, cond_x0, uncond_x0, timestep, x_t):
        scheduler = self.sample_scheduler
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        cond_eps = (x_t - alpha_prod_t ** 0.5 * cond_x0) / beta_prod_t ** 0.5
        uncond_eps = (x_t - alpha_prod_t ** 0.5 * uncond_x0) / beta_prod_t ** 0.5
        return cond_eps, uncond_eps

    def obtain_mu_t(self, x_0, timestep, x_t):

        scheduler = self.sample_scheduler
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = scheduler.alphas_cumprod[timestep-1]
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        mu_t = (((current_alpha_t **0.5) * (1-alpha_prod_t_prev)) * x_t) /(1-alpha_prod_t) + \
                (((alpha_prod_t_prev**0.5 * current_beta_t)/(1-alpha_prod_t)) * x_0)
        var_t = (1 - alpha_prod_t_prev) * current_beta_t / (1 - alpha_prod_t)
        return mu_t,var_t
