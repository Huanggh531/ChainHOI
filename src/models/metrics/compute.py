from typing import List

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric

from src.models.utils.tools import remove_padding
from src.transforms.joints2jfeats import Rifke
from src.models.utils.geometry import matrix_of_angles
from scipy.spatial.transform import Rotation
from .utils import l2_norm, variance,mean
import trimesh
import os
import numpy as np
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
class ComputeMetrics(Metric):

    def __init__(self,
                 njoints,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if jointstype not in ["mmm", "humanml3d"]:
            raise NotImplementedError("This jointstype is not implemented.")

        self.name = 'APE and AVE'
        self.jointstype = jointstype
        self.rifke = Rifke(jointstype=jointstype, normalization=False)

        self.force_in_meter = force_in_meter
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        # APE
        self.add_state("APE_root",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("APE_traj",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("APE_pose",
                       default=torch.zeros(njoints - 1),
                       dist_reduce_fx="sum")
        self.add_state("APE_joints",
                       default=torch.zeros(njoints),
                       dist_reduce_fx="sum")
        self.APE_metrics = ["APE_root", "APE_traj", "APE_pose", "APE_joints"]

        # AVE
        self.add_state("AVE_root",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("AVE_traj",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("AVE_pose",
                       default=torch.zeros(njoints - 1),
                       dist_reduce_fx="sum")
        self.add_state("AVE_joints",
                       default=torch.zeros(njoints),
                       dist_reduce_fx="sum")
        self.AVE_metrics = ["AVE_root", "AVE_traj", "AVE_pose", "AVE_joints"]

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics

    def compute(self, sanity_flag):
        count = self.count
        APE_metrics = {
            metric: getattr(self, metric) / count
            for metric in self.APE_metrics
        }

        # Compute average of APEs
        APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count

        # Remove arrays
        APE_metrics.pop("APE_pose")
        APE_metrics.pop("APE_joints")

        count_seq = self.count_seq
        AVE_metrics = {
            metric: getattr(self, metric) / count_seq
            for metric in self.AVE_metrics
        }

        # Compute average of AVEs
        AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq

        # Remove arrays
        AVE_metrics.pop("AVE_pose")
        AVE_metrics.pop("AVE_joints")

        return {**APE_metrics, **AVE_metrics}

    def update(self, jts_text: Tensor, jts_ref: Tensor, lengths: List[int]):
        torch.use_deterministic_algorithms(False)
        self.count += sum(lengths)#总的长度之和
        self.count_seq += len(lengths)#batch_size之和

        jts_text, poses_text, root_text, traj_text = self.transform(jts_text, lengths)

        jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref, lengths)

        for i in range(len(lengths)):
            self.APE_root += l2_norm(root_text[i], root_ref[i], dim=1).sum()
            self.APE_pose += l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
            self.APE_traj += l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
            self.APE_joints += l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)

            root_sigma_text = variance(root_text[i], lengths[i], dim=0)
            root_sigma_ref = variance(root_ref[i], lengths[i], dim=0)
            self.AVE_root += l2_norm(root_sigma_text, root_sigma_ref, dim=0)

            traj_sigma_text = variance(traj_text[i], lengths[i], dim=0)
            traj_sigma_ref = variance(traj_ref[i], lengths[i], dim=0)
            self.AVE_traj += l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

            poses_sigma_text = variance(poses_text[i], lengths[i], dim=0)
            poses_sigma_ref = variance(poses_ref[i], lengths[i], dim=0)
            self.AVE_pose += l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

            jts_sigma_text = variance(jts_text[i], lengths[i], dim=0)
            jts_sigma_ref = variance(jts_ref[i], lengths[i], dim=0)
            self.AVE_joints += l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)
        torch.use_deterministic_algorithms(True)

    def transform(self, joints: Tensor, lengths):

        features = self.rifke(joints)

        ret = self.rifke.extract(features)
        root_y, poses_features, vel_angles, vel_trajectory_local = ret

        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the local poses
        poses_local = rearrange(poses_features,
                                "... (joints xyz) -> ... joints xyz",
                                xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 2]],
                             rotations)
        poses = torch.stack(
            (poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local,
                                      rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # get the root joint
        root = torch.cat(
            (trajectory[..., :, [0]], root_y[..., None], trajectory[..., :,
                                                                    [1]]),
            dim=-1)

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)
        # put back the root joint y
        poses[..., 0, 1] = root_y

        # Add the trajectory globally
        poses[..., [0, 2]] += trajectory[..., None, :]

        if self.force_in_meter:
            # different jointstypes have different scale factors
            if self.jointstype == 'mmm':
                factor = 1000.0
            elif self.jointstype == 'humanml3d':
                factor = 1000.0 * 0.75 / 480.0
            # return results in meters
            return (remove_padding(poses / factor, lengths),
                    remove_padding(poses_local / factor, lengths),
                    remove_padding(root / factor, lengths),
                    remove_padding(trajectory / factor, lengths))
        else:
            return (remove_padding(poses, lengths),
                    remove_padding(poses_local,
                                   lengths), remove_padding(root, lengths),
                    remove_padding(trajectory, lengths))
  
from einops import repeat, rearrange
class ComputeMetrics_obj(Metric):

    def __init__(self,
                 njoints,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if jointstype not in ["mmm", "humanml3d"]:
            raise NotImplementedError("This jointstype is not implemented.")

        self.name = 'APE and AVE'
        self.jointstype = jointstype

        self.force_in_meter = force_in_meter
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        

        # APE
        self.add_state("APE_xyz",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("APE_rot",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("APE_position",
                       default=torch.zeros(512),
                       dist_reduce_fx="sum")
        self.APE_metrics = ["APE_xyz", "APE_rot","APE_position"]

        # AVE
        self.add_state("xyz",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("rot",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("position",
                       default=torch.zeros(512),
                       dist_reduce_fx="sum")
        self.add_state("AVE_xyz",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("AVE_rot",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("AVE_position",
                       default=torch.zeros(512),
                       dist_reduce_fx="sum")
        self.AVE_metrics = ["xyz", "rot","position","AVE_xyz", "AVE_rot","AVE_position"]

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics

    def compute(self, sanity_flag):
        count = self.count
        APE_metrics = {
            metric: getattr(self, metric) / count
            for metric in self.APE_metrics
        }
        # Compute average of APEs
        APE_metrics["APE_mean_xyz"] = self.APE_xyz.mean() / count
        APE_metrics["APE_mean_rot"] = self.APE_rot.mean() / count
        APE_metrics["APE_mean_position"] = self.APE_position.mean() / count
        # Remove arrays
        APE_metrics.pop("APE_xyz")
        APE_metrics.pop("APE_rot")
        APE_metrics.pop("APE_position")
        count_seq = self.count_seq
        AVE_metrics = {
            metric: getattr(self, metric) / count_seq
            for metric in self.AVE_metrics
        }

        # Compute average of AVEs
        AVE_metrics["xyz_mean"] = self.xyz.mean() / count_seq
        AVE_metrics["rot_mean"] = self.rot.mean() / count_seq
        AVE_metrics["position_mean"] = self.position.mean() / count_seq
        AVE_metrics["AVE_mean_xyz"] = self.AVE_xyz.mean() / count_seq
        AVE_metrics["AVE_mean_rot"] = self.AVE_rot.mean() / count_seq
        AVE_metrics["AVE_mean_position"] = self.AVE_position.mean() / count_seq
        # Remove arrays
        AVE_metrics.pop("xyz")
        AVE_metrics.pop("rot")
        AVE_metrics.pop("position")
        AVE_metrics.pop("AVE_xyz")
        AVE_metrics.pop("AVE_rot")
        AVE_metrics.pop("AVE_position")

        return {**APE_metrics, **AVE_metrics}

    def update(self, jts_text: Tensor, jts_ref: Tensor,obj_points:Tensor, obj_name,lengths: List[int]):
        
        

        self.count += sum(lengths)
        self.count_seq += len(lengths)
       
        

        
        #jts_text:[batch_size,L,6]
        xyz_text_t=jts_text[...,0:3]
        xyz_ref_t=jts_ref[...,0:3]
        rot_text_t=jts_text[...,3:]
        rot_ref_t=jts_ref[...,3:]
        
        
        objs_xyz=[]
        gt_objs_xyz=[]
        xyz_text=[]
        xyz_ref=[]
        rot_text=[]
        rot_ref=[]
        for i in range(len(lengths)):
            length = lengths[i]
            xyz_text.append(xyz_text_t[i,0:length,:])
            xyz_ref.append(xyz_ref_t[i,0:length,:])
            rot_text.append(rot_text_t[i,0:length,:])
            rot_ref.append(rot_ref_t[i,0:length,:])
            vertices = obj_points[i].cpu().numpy()#(batchsize,512,3)
            #mesh_path = os.path.join("./dataset/behave_t2m/object_mesh", simplified_mesh[obj_name[i].split('_')[2]])

            #temp_simp = trimesh.load(mesh_path)
            #all_vertices = temp_simp.vertices#(points，3)
            # center the meshes
            #center = np.mean(all_vertices, 0)
            #all_vertices -= center
            #new_vertices = np.concatenate([all_vertices, vertices[-2:]], 0)
            # transform
            
            motion_obj=(jts_text[i].cpu().numpy())[:length,:]  
            gt_motion_obj=(jts_ref[i].cpu().numpy())[:length,:]  
            #pred_obj_points
            angle, trans = motion_obj[:, :3].transpose(1,0), motion_obj[:, 3:].transpose(1,0)
            rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
            obj_point = np.matmul(vertices[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]
            objs_xyz.append(torch.tensor(obj_point).cuda())

            #gt_obj_points
            gt_angle, gt_trans = gt_motion_obj[:, :3].transpose(1,0), gt_motion_obj[:, 3:].transpose(1,0)
            gt_rot = Rotation.from_rotvec(gt_angle.transpose(1, 0)).as_matrix()
            gt_obj_point = np.matmul(vertices[np.newaxis], gt_rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + gt_trans.transpose(1, 0)[:, np.newaxis]
            gt_obj_point=torch.tensor(gt_obj_point)
            
            gt_objs_xyz.append(gt_obj_point.cuda())

        for i in range(len(lengths)):

            self.APE_xyz += l2_norm(xyz_text[i], xyz_ref[i], dim=1).sum()
            self.APE_rot += l2_norm(rot_text[i], rot_ref[i], dim=1).sum()
            self.APE_position+=l2_norm(objs_xyz[i], gt_objs_xyz[i], dim=2).sum(0)

            xyz_sigma_text = mean(xyz_text[i], lengths[i], dim=0)
            xyz_sigma_ref = mean(xyz_ref[i], lengths[i], dim=0)
            self.xyz += l2_norm(xyz_sigma_text, xyz_sigma_ref, dim=0)

            rot_sigma_text = mean(rot_text[i], lengths[i], dim=0)
            rot_sigma_ref = mean(rot_ref[i], lengths[i], dim=0)
            self.rot += l2_norm(rot_sigma_text, rot_sigma_ref, dim=0)
            
            obj_sigma_text = mean(objs_xyz[i], lengths[i], dim=0)
            obj_sigma_ref = mean(gt_objs_xyz[i], lengths[i], dim=0)
            self.position += l2_norm(obj_sigma_text, obj_sigma_ref, dim=1)

            xyz_sigma_text = variance(xyz_text[i], lengths[i], dim=0)
            xyz_sigma_ref = variance(xyz_ref[i], lengths[i], dim=0)
            self.AVE_xyz += l2_norm(xyz_sigma_text, xyz_sigma_ref, dim=0)

            rot_sigma_text = variance(rot_text[i], lengths[i], dim=0)
            rot_sigma_ref = variance(rot_ref[i], lengths[i], dim=0)
            self.AVE_rot += l2_norm(rot_sigma_text, rot_sigma_ref, dim=0)
            
            obj_sigma_text = variance(objs_xyz[i], lengths[i], dim=0)
            obj_sigma_ref = variance(gt_objs_xyz[i], lengths[i], dim=0)
            self.AVE_position += l2_norm(obj_sigma_text, obj_sigma_ref, dim=1)




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


#behave使用point_to_triangle_distance
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

import torch.nn.functional as F
#计算behave
class ContactMetric(Metric):

    def __init__(self,
                force_in_meter: bool = True,
                dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.force_in_meter=force_in_meter
        self.add_state("contact_distance",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#CD
        self.add_state("contact_distance_compair_gt",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#OCD
        self.add_state("skating_frames",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")
        self.add_state("total_ground_contact_frames",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")
        
        
        self.metrics = ["contact_distance","contact_distance_compair_gt","skating_frames","total_ground_contact_frames"]
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_acc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
     
    def compute(self, sanity_flag):
        count = self.count*22
        count_acc=self.count_acc
        count_seq=self.count_seq
        concat_metrics = {
            metric: getattr(self, metric) / count
            for metric in self.metrics
        }
        # Compute average of APEs

        concat_metrics["contact_distance_mean"] = self.contact_distance.mean()/ count_seq
        concat_metrics["contact_distance_compair_gt_mean"] = self.contact_distance_compair_gt.mean()/ count_seq
        concat_metrics["foot_skating_ratio_mean"] = self.skating_frames/ self.total_ground_contact_frames
    
        concat_metrics.pop("contact_distance")
        concat_metrics.pop("contact_distance_compair_gt")
        concat_metrics.pop("skating_frames")
        concat_metrics.pop("total_ground_contact_frames")
        return {**concat_metrics}

    def update(self, pred_pro: Tensor, gt_pro: Tensor,lengths: List[int],human_positions: Tensor,obj_point: Tensor,faces: List[int],name=None,dataset=None,idx=None,cal_type=2):
        self.count += sum(lengths)
        self.count_seq += len(lengths)
        #predictions = pred_pro > 0

        #predictions=torch.sigmoid(pred_pro) > 0.4
        contact_pro=gt_pro
        gt=gt_pro>0

        for b in range(len(lengths)):

            a = human_positions[b][:lengths[b], 10:12, :]  # shape: [T, 2, 3]
            # 提取 x（水平位置）和 y（垂直位置）
            x_positions = a[:, :, 0]  # shape: [T, 2]
            y_positions = a[:, :, 1]  # shape: [T, 2]

            # 检测地面接触 (y < 5 cm)
            ground_contact = y_positions < 0.05  # shape: [T, 2]
            # 计算相邻帧的 x 坐标位移 (取相邻时间帧之间的差异)
            x_displacement = torch.abs(x_positions[1:, :] - x_positions[:-1, :])  # shape: [T-1, 2]
            # 检测是否超过脚滑阈值 (2.5 cm)
            foot_skating = x_displacement > 0.025  # shape: [T-1, 2]
            # 忽略最后一帧的地面接触，因为要计算相邻帧
            ground_contact_reduced = ground_contact[1:, :]  # shape: [T-1, 2]
            # 只在地面接触期间检测脚滑
            foot_skating_during_contact = foot_skating & ground_contact_reduced  # shape: [T-1, 2]
            # 统计地面接触帧数
            total_ground_contact_frames = torch.sum(ground_contact_reduced)  # 标量, 总接触帧数
            # 统计发生脚滑的帧数
            skating_frames = torch.sum(foot_skating_during_contact)  # 标量, 脚滑帧数
            # 计算脚滑比率
            self.skating_frames += skating_frames
            self.total_ground_contact_frames += total_ground_contact_frames


            faces_expanded = faces[b].unsqueeze(0).unsqueeze(-1).expand(obj_point.shape[1], -1, -1, 3)  # [T, M, 3, 3]
            # 使用torch.gather获取对应的点
            obj_point_expanded = obj_point[b].unsqueeze(1).expand(-1, faces[b].shape[0], -1, -1)  # [T, M, N, 3]
            indexed_vertices = torch.gather(obj_point_expanded, 2, faces_expanded)
            distances=point_to_triangle_distance(human_positions[b:b+1],indexed_vertices.unsqueeze(0))[0]
            mse_loss = F.mse_loss(torch.zeros_like(distances), distances, reduction='none')
            contact_mask = (contact_pro[b] > 0).float()
            contact_mask[lengths[b]:]=0
            masked_loss = mse_loss * contact_mask
            if contact_mask.sum() > 0:
                self.contact_distance += masked_loss.sum() / contact_mask.sum()

            ##和gt对比选取contact label
            if cal_type == 1:
                obj_name=name[b]
                contact_new = None
                dis_min = 100000
                min_frame_b = 0
                if len(dataset.text_split[obj_name])==1:
                    if contact_mask.sum() > 0:
                        self.contact_distance_compair_gt +=  masked_loss.sum() / contact_mask.sum()
                else:
                    for k in range(len(dataset.text_split[obj_name])):
                        com_name = dataset.text_split[obj_name][k]
                        if com_name not in dataset.name_gtmotion.keys() or com_name not in dataset.name_contact_label.keys():
                            continue
                        com_human = torch.tensor(dataset.name_gtmotion[com_name][0][idx[b]:idx[b]+lengths[b]]).to(human_positions.device)
                        if abs(com_human.shape[0]-human_positions[b][:lengths[b]].shape[0])<=8:
                            min_frame = min(com_human.shape[0],human_positions[b][:lengths[b]].shape[0])
                            dis = F.mse_loss(com_human[:min_frame],human_positions[b][:lengths[b]][:min_frame])
                            if contact_new is None or dis_min > dis:
                                min_frame_b = min_frame
                                dis_min = dis
                                contact_new = torch.tensor(dataset.name_contact_label[com_name][idx[b]:idx[b]+lengths[b]])[:min_frame].to(human_positions.device)

                    contact_mask2 = (contact_pro[b] > 0).float()
                    contact_mask2[min_frame_b:]=0
                    contact_mask2[:min_frame_b] = (contact_new > 0).float()
                    masked_loss2 = mse_loss * contact_mask2
                    if contact_mask2.sum() > 0:
                        self.contact_distance_compair_gt += masked_loss2.sum() / contact_mask2.sum()
            else:
                obj_name=name[b]
                dis_min = 100000
                if contact_mask.sum() > 0:
                    dis_min = masked_loss.sum() / contact_mask.sum()
                
                if obj_name not in dataset.text_split.keys() or len(dataset.text_split[obj_name])==1:
                    if contact_mask.sum() > 0:
                        self.contact_distance_compair_gt +=  masked_loss.sum() / contact_mask.sum()
                else:
                    for k in range(len(dataset.text_split[obj_name])):
                        contact_new = None
                        com_name = dataset.text_split[obj_name][k]
                        if com_name not in dataset.name_gtmotion.keys() or com_name not in dataset.name_contact_label.keys():
                            continue
                        com_human = torch.tensor(dataset.name_gtmotion[com_name][0][idx[b]:idx[b]+lengths[b]]).to(human_positions.device)
                        if abs(com_human.shape[0]-human_positions[b][:lengths[b]].shape[0])<=8:
                            min_frame = min(com_human.shape[0],human_positions[b][:lengths[b]].shape[0])
                            contact_new = torch.tensor(dataset.name_contact_label[com_name][idx[b]:idx[b]+lengths[b]])[:min_frame].to(human_positions.device)
                            contact_mask2 = (contact_pro[b] > 0).float()
                            contact_mask2[min_frame:]=0
                            contact_mask2[:min_frame] = (contact_new > 0).float()
                            masked_loss2 = mse_loss * contact_mask2
                            if contact_mask2.sum() > 0:
                                if dis_min > masked_loss2.sum() / contact_mask2.sum():
                                    dis_min = masked_loss2.sum() / contact_mask2.sum()
                    if  dis_min != 100000 :    
                        self.contact_distance_compair_gt += dis_min
            

#计算behave的gt接触
class ContactMetricGT(Metric):

    def __init__(self,
                force_in_meter: bool = True,
                dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.force_in_meter=force_in_meter
        self.add_state("contact_distance",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#评估有接触的接触距离
        self.add_state("pene_distance",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#穿模距离
        self.add_state("pene_num_joints",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#穿模距离
        
        
        self.metrics = ["contact_distance","pene_distance","pene_num_joints"]
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_acc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
     
    def compute(self, sanity_flag):
        count = self.count*22
        count_acc=self.count_acc
        count_seq=self.count_seq
        concat_metrics = {
            metric: getattr(self, metric) / count
            for metric in self.metrics
        }
        # Compute average of APEs

        concat_metrics["contact_distance_mean"] = self.contact_distance.mean()/ count_seq
        concat_metrics["pene_distance_mean"] = self.pene_distance.mean()/ count_seq
        concat_metrics["pene_num_joints_mean"] = self.pene_num_joints.mean()/ count_seq
        concat_metrics.pop("contact_distance")
        concat_metrics.pop("pene_distance")
        concat_metrics.pop("pene_num_joints")
        return {**concat_metrics}

    def update(self, pred_pro: Tensor, gt_pro: Tensor,lengths: List[int],human_positions: Tensor,obj_point: Tensor,faces: List[int],contact: Tensor):
        self.count += sum(lengths)
        self.count_seq += len(lengths)
        predictions = pred_pro > 0
        contact_pro=gt_pro
        gt=gt_pro>0
        
        
        for b in range(len(lengths)):
            faces_expanded = faces[b].unsqueeze(0).unsqueeze(-1).expand(obj_point.shape[1], -1, -1, 3)  # [T, M, 3, 3]
            # 使用torch.gather获取对应的点
            obj_point_expanded = obj_point[b].unsqueeze(1).expand(-1, faces[b].shape[0], -1, -1)  # [T, M, N, 3]
            indexed_vertices = torch.gather(obj_point_expanded, 2, faces_expanded)
            distances=point_to_triangle_distance(human_positions[b:b+1],indexed_vertices.unsqueeze(0))[0]
            mse_loss = F.l1_loss(torch.zeros_like(distances), distances, reduction='none')
            contact_mask = (contact_pro[b] > 0).float()
            contact_mask[lengths[b]:]=0
            masked_loss = mse_loss * contact_mask
           
            #import pdb
            #pdb.set_trace()
            
            if contact_mask.sum() > 0:
                self.contact_distance += masked_loss.sum() / contact_mask.sum()
            # 计算小于零的值的平均值
            sdf_loss=distances[:lengths[b]]
            negative_values = sdf_loss[sdf_loss < 0]  # 获取小于零的值
            self.pene_num_joints += negative_values.numel()/(sdf_loss.shape[0]*sdf_loss.shape[1])
            self.pene_distance += (-1 * torch.mean(negative_values)) if negative_values.numel() > 0 else torch.tensor(0.0)  # 计算平均值   








#omomo使用point_to_triangle_distance3
def point_to_triangle_distance3(points, triangles):
    
    # print(points)
    # points [B, N, 3] triangles [B, M, 3, 3]
    B, N, M = points.shape[0], points.shape[1], triangles.shape[1]
    v0, v1, v2 = triangles[:, :, 0], triangles[:, :, 1], triangles[:, :, 2]
    # 边向量和法线向量
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = torch.cross(edge1, edge2)

    eps = 1e-8  # 设置一个很小的值 epsilon，用来避免除以 0
    # 计算每个 normal 向量的模长
    normal_norm = torch.norm(normal, dim=-1, keepdim=True)
    # 防止模长为 0 的情况，在模长上加上 epsilon
    normal = normal / (normal_norm + eps)
    #normal = normal / torch.norm(normal, dim=-1, keepdim=True) # B M 3
    
    n_points = repeat(points, "b n d -> b n m d", m=M)
    n_norms = repeat(normal, "b m d -> b n m d", n=N)
    n_v0 = repeat(v0, "b m d -> b n m d", n=N)
    
    to_v0 = n_points - n_v0 # B N M 3
    
    ori_dist_to_plan = torch.einsum("bnmd,bnmd->bnm", to_v0, n_norms)
    dist_to_plane = ori_dist_to_plan
    # print(normal)
    # print(dist_to_plane)
    # print(ori_dist_to_plan)
    
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
    

    def closest_point_on_edge_2(points, v0, v1, v2):
        # points [B, N, M, 3] v0, v1, v2 [B, M, 3]
        B, N, M = points.shape[:3]
       
        
        edges = rearrange([v1 - v0, v2 - v0, v2 - v1], "K B M D -> B M K D")
        point_to_edge = repeat(points, "B N M D -> B N M I D", I=1) - rearrange([v0, v1, v1], "K B M D -> B M K D")

        #t = torch.einsum("bnmkd,bmkd->bnmk", point_to_edge, edges) / torch.einsum("bmkd,bmkd->bmk", edges, edges).unsqueeze(dim=1)
        # 计算 point_to_edge 与 edges 的内积
        numerator = torch.einsum("bnmkd,bmkd->bnmk", point_to_edge, edges)

        # 计算 edges 的模长平方
        # 避免除以零，加入 epsilon
        eps = 1e-7

        denominator = torch.einsum("bmkd,bmkd->bmk", edges, edges)
        denominator = torch.where(denominator == 0, torch.tensor(eps, device=denominator.device), denominator)
        denominator = denominator.unsqueeze(dim=1)
        t = torch.einsum("bnmkd,bmkd->bnmk", point_to_edge, edges) / denominator
        
        t = torch.clamp(t, 0, 1) # b n m k
        closest_point = t.unsqueeze(dim=-1) * edges.view([B, 1, M, 3, -1]) + repeat([v0, v0, v1], "K B M D -> B N M K D", N=1)
        dist = (points.unsqueeze(dim=3) - closest_point).norm(dim=-1)
        return dist.min(dim=-1)[0]

    u, v, w = barycentric_coords(projection, n_v0, v1.unsqueeze(dim=1), v2.unsqueeze(dim=1))
    
    # 判断投影是否在三角形内部，不在内部则置为inf
    inside = (u >= 0) & (v >= 0) & (w >= 0)
    distances_to_edges = torch.where(~inside, torch.tensor(float('inf'), device=points.device), dist_to_plane)
    # print(distances_to_edges)
    
    # 计算点到三角面片上的三条边上的最小距离
    dist_1 = closest_point_on_edge_2(n_points, v0, v1, v2)
    # 加上距离正负号，
    dist_1 = dist_1 * torch.sign(ori_dist_to_plan+1e-7)

    # 选取绝对值最小的距离 -> 离三角面片最近
    dist_1 = torch.gather(dist_1, -1, dist_1.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    dist_2 = torch.gather(distances_to_edges, -1, distances_to_edges.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    stack_dist = torch.stack([dist_1, dist_2], dim=-1)
    dist = torch.gather(stack_dist, -1, stack_dist.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    # print(dist_to_edges)
    
    return dist

@torch.no_grad()
def closest_point_on_edge2(points, v0, v1, v2):
    # points [B, T, N, M, 3] v0, v1, v2 [B, T, M, 3]
    B, T, N, M = points.shape[:4]
    edges = rearrange([v1 - v0, v2 - v0, v2 - v1], "K B T M D -> B T M K D")
    point_to_edge = repeat(points, "B T N M D -> B T N M K D", K=3) - repeat(rearrange([v0, v1, v1], "K B T M D -> B T M K D"),"B T M K D -> B T N M K D",N=22)
    
    eps = 1e-6
    #numerator = torch.einsum("btnmkd,btmkd->btnmk", point_to_edge, edges)
    denominator = torch.einsum("btmkd,btmkd->btmk", edges, edges)
    denominator = torch.where(denominator == 0, torch.tensor(eps, device=denominator.device), denominator)
    denominator = denominator.unsqueeze(dim=2)

    t = torch.einsum("btnmkd,btmkd->btnmk", point_to_edge, edges) / denominator
    t = torch.clamp(t, 0, 1) # b t n m k
    closest_point = t.unsqueeze(dim=-1) * edges.unsqueeze(2) + rearrange([v0, v0, v1], "K B T M D -> B T M K D").unsqueeze(2)
    dist = (points.unsqueeze(dim=4) - closest_point).norm(dim=-1)
    return dist.min(dim=-1)[0]

def point_to_triangle_distance2(points, triangles):

    # points [B, T, N, 3] triangles [B, T, M, 3, 3]
    B, T, N, M = points.shape[0], points.shape[1], points.shape[2], triangles.shape[2]
    v0, v1, v2 = triangles[:, :, :, 0], triangles[:, :, :, 1], triangles[:, :, :, 2]
    # 边向量和法线向量
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = torch.cross(edge1, edge2)

    eps = 1e-8  # 设置一个很小的值 epsilon，用来避免除以 0
    # 计算每个 normal 向量的模长
    normal_norm = torch.norm(normal, dim=-1, keepdim=True)
    # 防止模长为 0 的情况，在模长上加上 epsilon
    normal = normal / (normal_norm + eps)

    #normal = normal / torch.norm(normal, dim=-1, keepdim=True) # B T M 3

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
    dist_1 = closest_point_on_edge2(n_points, v0, v1, v2)
    # 加上距离正负号
    dist_1 = dist_1 * torch.sign(ori_dist_to_plan + 1e-8)
    
    # 选取绝对值最小的距离 -> 离三角面片最近
    dist_1 = torch.gather(dist_1, -1, dist_1.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    dist_2 = torch.gather(distances_to_edges, -1, distances_to_edges.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    stack_dist = torch.stack([dist_1, dist_2], dim=-1)
    dist = torch.gather(stack_dist, -1, stack_dist.abs().argmin(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)
    
    return dist

#计算omomo
from torchsdf import index_vertices_by_faces, compute_sdf
class ContactMetric2(Metric):

    def __init__(self,
                force_in_meter: bool = True,
                dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.force_in_meter=force_in_meter
        self.add_state("contact_distance",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#评估有接触的接触距离
        self.add_state("contact_distance2",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#评估有接触的接触距离
        self.add_state("contact_distance_compair_gt",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#评估有接触的接触距离
                        
        self.add_state("pene_distance",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#穿模距离
        self.add_state("pene_num_joints",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#穿模距离
        self.add_state("pene_num_joints2",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#穿模距离
        self.add_state("foot_skating_ratio",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#穿模距离
        self.add_state("skating_frames",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")
        self.add_state("total_ground_contact_frames",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")
        
        self.metrics = ["contact_distance","contact_distance_compair_gt","pene_distance","pene_num_joints","foot_skating_ratio","pene_num_joints2","contact_distance2","skating_frames","total_ground_contact_frames"]
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_acc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
     
    def compute(self, sanity_flag):
        count = self.count*8
        count_acc=self.count_acc
        count_seq=self.count_seq
        concat_metrics = {
            metric: getattr(self, metric) / count
            for metric in self.metrics
        }
        # Compute average of APEs

        concat_metrics["contact_distance_mean"] = self.contact_distance.mean()/ count_seq
        concat_metrics["contact_distance_compair_gt_mean"] = self.contact_distance_compair_gt.mean()/ count_seq
        concat_metrics["pene_distance_mean"] = self.pene_distance.mean()/ count_seq
        concat_metrics["pene_num_joints_mean"] = self.pene_num_joints.mean()/ count_seq
        concat_metrics["foot_skating_ratio_mean"] = self.foot_skating_ratio.mean()/ count_seq
        concat_metrics["pene_num_joints2_mean"] = self.pene_num_joints2.mean()/ count_seq
        concat_metrics["contact_distance2_mean"] = self.contact_distance2.mean()/ count_seq
        concat_metrics["foot_skating_ratio_mean2"] = self.skating_frames/ self.total_ground_contact_frames
        concat_metrics.pop("skating_frames")
        concat_metrics.pop("total_ground_contact_frames")
        concat_metrics.pop("contact_distance")
        concat_metrics.pop("contact_distance_compair_gt")
        concat_metrics.pop("pene_distance")
        concat_metrics.pop("pene_num_joints")
        concat_metrics.pop("foot_skating_ratio")
        concat_metrics.pop("pene_num_joints2")
        concat_metrics.pop("contact_distance2")
        return {**concat_metrics}
    def update(self,gt_pro: Tensor,lengths: List[int],human_positions: Tensor,obj_point: Tensor,faces: List[int],contact:Tensor,name=None,dataset=None,idx=None,cal_type=2):
        self.count += sum(lengths)
        self.count_seq += len(lengths)
        contact_pro=gt_pro
        for b in range(len(lengths)):
            a = human_positions[b][:lengths[b], 10:12, :]  # shape: [T, 2, 3]
            # 提取 x（水平位置）和 y（垂直位置）
            # floor_height = a.min(axis=0).min(axis=0)[2]
            # a[:, :, 2] += floor_height
            
            # 第一次沿 axis=0 求最小值
            min_vals, _ = torch.min(a, dim=0)
            # 第二次沿 axis=0 求最小值
            min_vals_2, _ = torch.min(min_vals, dim=0)
            # 取索引为1的最小值
            floor_height = min_vals_2[1]

            a[:, :, 1] -= floor_height
            x_positions = a[:, :, 0]  # shape: [T, 2]
            y_positions = a[:, :, 1]  # shape: [T, 2]
            
            # 检测地面接触 (y < 5 cm)
            
            ground_contact = y_positions < 0.05  # shape: [T, 2]
            # 计算相邻帧的 x 坐标位移 (取相邻时间帧之间的差异)
            x_displacement = torch.abs(x_positions[1:, :] - x_positions[:-1, :])  # shape: [T-1, 2]
            # 检测是否超过脚滑阈值 (2.5 cm)
            foot_skating = x_displacement > 0.025  # shape: [T-1, 2]
            # 忽略最后一帧的地面接触，因为要计算相邻帧
            ground_contact_reduced = ground_contact[1:, :]  # shape: [T-1, 2]
            # 只在地面接触期间检测脚滑
            foot_skating_during_contact = foot_skating & ground_contact_reduced  # shape: [T-1, 2]
            # 统计地面接触帧数
            total_ground_contact_frames = torch.sum(ground_contact_reduced)  # 标量, 总接触帧数
            # 统计发生脚滑的帧数
            skating_frames = torch.sum(foot_skating_during_contact)  # 标量, 脚滑帧数
            # 计算脚滑比率
            self.skating_frames += skating_frames
            self.total_ground_contact_frames += total_ground_contact_frames
            if total_ground_contact_frames > 0:
                foot_skating_ratio = skating_frames / total_ground_contact_frames
                self.foot_skating_ratio += foot_skating_ratio

            
            len_=lengths[b]
            distances_ts=[]

            for k in range(len_): 
                
                indexed_vertices = index_vertices_by_faces(obj_point[b][k], faces[b].to(human_positions.device))
                human_positions_batch=human_positions[b]
                distances_ts_ = point_to_triangle_distance3(human_positions_batch[k].unsqueeze(0), indexed_vertices.unsqueeze(0))
                distances_ts.append(distances_ts_)
            distances=torch.stack(distances_ts).squeeze(1)
            contact_b=contact[b][:len_].to(human_positions.device)
            

            # faces_expanded = faces[b].unsqueeze(0).unsqueeze(-1).expand(obj_point.shape[1], -1, -1, 3)  # [T, M, 3, 3]
            # # 使用torch.gather获取对应的点
            # obj_point_expanded = obj_point[b].unsqueeze(1).expand(-1, faces[b].shape[0], -1, -1)  # [T, M, N, 3]
            # indexed_vertices = torch.gather(obj_point_expanded, 2, faces_expanded)
            # distances2=point_to_triangle_distance2(human_positions[b:b+1],indexed_vertices.unsqueeze(0))[0]
            # mse_loss2 = F.mse_loss(torch.zeros_like(distances2), distances2, reduction='none')

            #contact_point=[0,10,11,12,16,17,20,21]
            #mse_loss=mse_loss[...,contact_point]
            
            # 计算布尔张量，判断 distances 是否大于 contact
            greater_than_contact = distances > contact_b
            # 计算布尔张量，判断 distances 是否小于 contact
            less_than_contact = distances < contact_b

            # 只取第 20 和第 21 个关节点（即索引 19 和 20）的布尔值
            selected_greater_than_contact = greater_than_contact[:, [20, 21]]

            # 计算满足条件的数量 (True 转换为 1，False 转换为 0)
            count_greater_than_contact = selected_greater_than_contact.sum()
            self.contact_distance2 += count_greater_than_contact / selected_greater_than_contact.numel()
            # 计算总数量
            total_elements = less_than_contact.sum()
            self.pene_num_joints2 += total_elements / less_than_contact.numel()

            
            #faces_expanded = faces[b].unsqueeze(0).unsqueeze(-1).expand(obj_point.shape[1], -1, -1, 3).cuda()  # [T, M, 3, 3]
            # 使用torch.gather获取对应的点
            #obj_point_expanded = obj_point[b].unsqueeze(1).expand(-1, faces[b].shape[0], -1, -1)  # [T, M, N, 3]
            #indexed_vertices = torch.gather(obj_point_expanded, 2, faces_expanded)
            #point_to_triangle_distance(human_positions[0][0:1],indexed_vertices[0].unsqueeze(0))[0]
            #distances=point_to_triangle_distance(human_positions[b:b+1],indexed_vertices.unsqueeze(0))[0]
            contact_dis = distances

            mse_loss = F.mse_loss(torch.zeros_like(contact_dis), contact_dis, reduction='none')[0:lengths[b]]
            contact_mask=torch.ones(mse_loss.shape[0],mse_loss.shape[1],device=mse_loss.device)

            contact_mask = (contact_pro[b,0:lengths[b]] > 0).float()
            contact_mask[:,0:20] = 0
            masked_loss = mse_loss * contact_mask
            if contact_mask.sum() > 0:
                self.contact_distance += masked_loss.sum() / contact_mask.sum()


            if cal_type == 1:
                ##和gt对比选取contact label
                obj_name=name[b]
                contact_new = None
                dis_min = 100000
                min_frame_b = 0
                if obj_name not in dataset.text_split.keys() or len(dataset.text_split[obj_name])==1:
                    if contact_mask.sum() > 0:
                        self.contact_distance_compair_gt +=  masked_loss.sum() / contact_mask.sum()
                else:
                    for k in range(len(dataset.text_split[obj_name])):
                        com_name = dataset.text_split[obj_name][k]
                        if com_name not in dataset.name_gtmotion.keys():
                            continue
                        com_human = torch.tensor(dataset.name_gtmotion[com_name][0][idx[b]:idx[b]+lengths[b]]).to(human_positions.device)
                        if abs(com_human.shape[0]-human_positions[b][:lengths[b]].shape[0])<=8:
                            min_frame = min(com_human.shape[0],human_positions[b][:lengths[b]].shape[0])
                            dis = F.mse_loss(com_human[:min_frame],human_positions[b][:lengths[b]][:min_frame])
                            if contact_new is None or dis_min > dis:
                                min_frame_b = min_frame
                                dis_min = dis
                                contact_new = torch.tensor(dataset.name_contact_label[com_name][idx[b]:idx[b]+lengths[b]])[:min_frame].to(human_positions.device)

                    contact_mask2 = (contact_pro[b,0:lengths[b]] > 0).float()
                    
                    contact_mask2[min_frame_b:]=0
                    contact_mask2[:min_frame_b] = (contact_new > 0).float()
                    contact_mask2[:,0:20]=0
                    masked_loss2 = mse_loss * contact_mask2
                    if contact_mask2.sum() > 0:
                        self.contact_distance_compair_gt += masked_loss2.sum() / contact_mask2.sum()
            else:
                ##和gt对比选取contact label
                obj_name=name[b]
                dis_min = 100000
                if contact_mask.sum() > 0:
                    dis_min = masked_loss.sum() / contact_mask.sum()
                if obj_name not in dataset.text_split.keys() or len(dataset.text_split[obj_name])==1:
                    if contact_mask.sum() > 0:
                        self.contact_distance_compair_gt +=  masked_loss.sum() / contact_mask.sum()
                else:
                    for k in range(len(dataset.text_split[obj_name])):
                        contact_new = None
                        com_name = dataset.text_split[obj_name][k]
                        if com_name not in dataset.name_gtmotion.keys():
                            continue
                        com_human = torch.tensor(dataset.name_gtmotion[com_name][0][idx[b]:idx[b]+lengths[b]]).to(human_positions.device)
                        if abs(com_human.shape[0]-human_positions[b][:lengths[b]].shape[0])<=8:
                            min_frame = min(com_human.shape[0],human_positions[b][:lengths[b]].shape[0])
                            contact_new = torch.tensor(dataset.name_contact_label[com_name][idx[b]:idx[b]+lengths[b]])[:min_frame].to(human_positions.device)
                            contact_mask2 = (contact_pro[b,0:lengths[b]] > 0).float()
                            
                            contact_mask2[min_frame:]=0
                            contact_mask2[:min_frame] = (contact_new > 0).float()
                            contact_mask2[:,0:20]=0
                            masked_loss2 = mse_loss * contact_mask2
                            if contact_mask2.sum() > 0:
                                if dis_min > masked_loss2.sum() / contact_mask2.sum():
                                    dis_min = masked_loss2.sum() / contact_mask2.sum()
                    if dis_min != 100000:
                        self.contact_distance_compair_gt += dis_min

            # 计算小于零的值的平均值
            sdf_loss = distances[:lengths[b]]
            negative_values = sdf_loss[sdf_loss < 0]  # 获取小于零的值
            self.pene_num_joints += negative_values.numel()/(sdf_loss.shape[0]*sdf_loss.shape[1])
            self.pene_distance += (-1 * torch.mean(negative_values)) if negative_values.numel() > 0 else torch.tensor(0.0)  # 计算平均值
            
from visualize.utils.rotation2xyz import Rotation2xyz
from visualize.utils.simplify_loc2rot import joints2smpl
class PeneMetric(Metric):

    def __init__(self,
                force_in_meter: bool = True,
                dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.force_in_meter=force_in_meter
        
        self.add_state("pene_num_mesh",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#穿模数量
        self.add_state("total_pene_num_mesh",
                       default=torch.tensor(0.),
                        dist_reduce_fx="sum")#穿模距离
        
        
        self.metrics = ["pene_num_mesh","total_pene_num_mesh"]
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_acc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
     
    def compute(self, sanity_flag):
        count = self.count*22
        count_acc=self.count_acc
        count_seq=self.count_seq
        concat_metrics = {
            metric: getattr(self, metric) / count
            for metric in self.metrics
        }
        # Compute average of APEs

        
        concat_metrics["pene_num_mesh_mean"] = self.pene_num_mesh/ self.total_pene_num_mesh
    

        concat_metrics.pop("pene_num_mesh")
        concat_metrics.pop("total_pene_num_mesh")
        return {**concat_metrics}

    def update(self, pred_pro: Tensor, gt_pro: Tensor,lengths: List[int],human_positions: Tensor,obj_point: Tensor,faces: List[int]):
        self.count += sum(lengths)
        self.count_seq += len(lengths)
        #predictions = pred_pro > 0

        #predictions=torch.sigmoid(pred_pro) > 0.4
        contact_pro=gt_pro
        gt=gt_pro>0
        # 将预测结果和目标标签相比较
        # for k in np.arange(0.1,0.5,0.01):
        #     predictions=torch.sigmoid(pred_pro) > k
            # self.only_concat=0
            # self.acc_concat=0
            # self.count_acc=0 


        for b in range(human_positions.shape[0]):
            motions = human_positions[b][:lengths[b]]
            frames, njoints, nfeats = motions.shape
            min_values, _ = motions.min(dim=0)  # 获取第一个维度的最小值
            MINS = min_values.min(dim=0)[0]  

            height_offset = MINS[1]
            motions[:, :, 1] -= height_offset

            rot2xyz = Rotation2xyz(device=motions.device)
            faces2 = rot2xyz.smpl_model.faces
            j2s = joints2smpl(num_frames=frames, device_id=motions.device.index, cuda=True)
            print(f'Running SMPLify, it may take a few minutes.')
            with torch.set_grad_enabled(True):
                motion_tensor, opt_dict = j2s.joint2smpl(motions.cpu().numpy())  # [nframes, njoints, 3] for hml3d
            vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                            pose_rep='rot6d', translation=True, glob=True,
                            jointstype='vertices',
                            vertstrans=True).cpu()
            
            vertices = torch.tensor(vertices,dtype=motions.dtype).to(motions.device)
            #vertices = torch.tensor(np.load("/home/guohong/HOI/result/vertices.npy"),dtype=motions.dtype).to(motions.device)
            obj_point_t = obj_point[b][:lengths[b]]
            # faces_expanded = faces[b].unsqueeze(0).unsqueeze(-1).expand(obj_point_t.shape[0], -1, -1, 3)  # [T, M, 3, 3]
            # # 使用torch.gather获取对应的点
            # obj_point_expanded = obj_point_t.unsqueeze(1).expand(-1, faces[b].shape[0], -1, -1)  # [T, M, N, 3]
            # indexed_vertices = torch.gather(obj_point_expanded, 2, faces_expanded)
            vertices = rearrange(vertices,"b n m t -> b t n m").contiguous()

            from torchsdf import index_vertices_by_faces, compute_sdf
            #distances=point_to_triangle_distance(vertices,indexed_vertices.unsqueeze(0))[0]
            distances=[]
            for t in range(vertices[0].shape[0]):
                indexed_vertices = index_vertices_by_faces(obj_point_t[t], faces[b])
                distances_ts2,sign,_,_ = compute_sdf(vertices[0][t], indexed_vertices)
                distances_ts2 = distances_ts2 * sign
                distances.append(distances_ts2)
            distances = torch.stack(distances)
            negative_values = distances[distances < 0] 

            self.pene_num_mesh += negative_values.numel()
            self.total_pene_num_mesh += distances.shape[0] * distances.shape[1]
            

            