import torch
from src.data.humanml.scripts.motion_process import recover_from_ric
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

from trimesh import Trimesh
import trimesh
import math

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


class Guide_Contact:
    def __init__(self,
                 inv_transform_th=None,
                 classifiler_scale=10.0,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 use_global=False,
                 batch_size=10,
                 afford_sample=None,
                 mean=None,
                 std=None
                 ):

        self.classifiler_scale = classifiler_scale
        self.inv_transform_th = inv_transform_th 
        self.n_joints = 22
        self.sigmoid = torch.nn.Sigmoid()
        self.mean=mean
        self.std=std
        self.use_global = use_global
        self.batch_size = batch_size


        self.afford_sample = afford_sample

        self.loss_all = []


    def __call__(self, x, t, obj_points=None, human_mean=None): # *args, **kwds):
        """
        Args:
            target: [bs, 120, 22, 3]
            target_mask: [bs, 120, 22, 3]
        """

            
        # return x.detach()
        
        loss, grad, loss_list = self.gradients(x, t, self.afford_sample, obj_points, None)

            
        return loss, grad, loss_list

    def gradients(self, x, t, afford_sample, obj_points, obj_normals):
        
        with torch.enable_grad():
            
            n_joints = 22 
            x.requires_grad_(True)
            sample = x.permute(0, 2, 3, 1) * torch.from_numpy(self.std).to(x.device) + torch.from_numpy(self.mean).to(x.device)
            B, _, T , _ = sample.shape
            sample_obj = sample[..., 263:]
            sample_obj = sample_obj.permute(0, 1, 3, 2)
            sample = recover_from_ric(sample.float(), n_joints)
            sample = sample[:,:,:,:n_joints*3]
            joints_output = sample.reshape(sample.shape[0], sample.shape[2], n_joints, 3)

            obj_output = sample_obj[:,0,:,:].permute(0,2,1).float()


            contact_idxs = []
            o_afford_labels = []

            for i in range(afford_sample.shape[0]):
                contact_prob = afford_sample[i,3:,0, :].permute(1,0)
                contact_pos = afford_sample[i,:3, 0, :].permute(1,0)
                contact_idx = torch.where(contact_prob>0.65)[0]
                points = obj_points[i]
                if len(contact_idx)>0:
                    sel_pos = contact_pos[contact_idx].to(points.device)                    
                    dist = torch.cdist(sel_pos, points)
                    min_dist_idx = torch.argmin(dist, dim=-1)
                    o_afford_labels.append(min_dist_idx.detach().cpu().numpy())
                    contact_idxs.append(contact_idx.detach().cpu().numpy())
                else:
                    o_afford_labels.append(np.array([-1]))
                    contact_idxs.append(np.array([-1]))
            batch_size = joints_output.size(0)
            all_loss_joints_contact = 0
            all_loss_object_contact = 0
            contact_loss= torch.zeros(0).to(x.device)
            all_loss_static = torch.zeros(0).to(x.device)
            all_loss_static_xz = torch.zeros(0).to(x.device)
            all_local_rot = torch.zeros(0).to(x.device)
            all_close_points_loss = torch.zeros(0).to(x.device)


            for i in range(B):      
                # center
                vertices = obj_points[i][:-2,:].float()
                center = torch.mean(vertices, 0)
                vertices = vertices - center
                center_ = torch.mean(vertices, 0)

                init_y = center_[1:2] - vertices[:, 1].min()

                contact_vertices = obj_points[i][-2:,:].float()


                #obj_normal = obj_normals[i]

                pred_angle, pred_trans = obj_output[i, :, :3].transpose(1,0), obj_output[i, :, 3:].transpose(1,0)
                pred_rot = axis_angle_to_matrix(pred_angle.transpose(1,0))

                pred_points = torch.matmul(contact_vertices.unsqueeze(0), pred_rot.permute(0, 2, 1)) + pred_trans.transpose(1, 0).unsqueeze(1)

                all_pred_points = torch.matmul(obj_points[i].float().unsqueeze(0), pred_rot.permute(0, 2, 1)) + pred_trans.transpose(1, 0).unsqueeze(1)

                if contact_idxs[i].any() !=-1:
                    # sel_joints = np.array([0,9,10,11,16,17,20,21])
                    # contact_idxs[i] = np.array([6, 7])
                    sel_joints = np.array([20,21])
                    contact_idxs[i] = np.array([0, 1])
                    o_afford_labels[i] = o_afford_labels[i][:2]

                    sel_idx = sel_joints[contact_idxs[i]]
                    loss_contact = torch.norm((joints_output[i, :, sel_idx,:] - all_pred_points[:, o_afford_labels[i],  :]), dim=-1)
                    contact_loss = torch.cat([contact_loss, loss_contact.sum(-1).unsqueeze(0)], dim=0)

            
            total_loss_contact = 1.0 * contact_loss.sum()
            loss_sum = total_loss_contact
            self.loss_all.append(loss_sum)
            grad = torch.autograd.grad([loss_sum], [x])[0]
            x.detach()
        return loss_sum, grad, self.loss_all

