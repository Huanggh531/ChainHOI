import os
import os.path as osp
import numpy as np
import torch

hml3d_root_joint_l, hml3d_root_joint_r = 0, 4
hml3d_local_joint_pos_l, hml3d_local_joint_pos_r = 4, 67  # 4, 4+21*3
hml3d_joint_rotation_l, hml3d_joint_rotation_r = 67, 193  # 4+21*3, 4+21*3+21*6
hml3d_joint_velocity_l, hml3d_joint_velocity_r = 193, 259  # 4+21*3+21*6+22*3, -4
hml3d_foot_contacts_l, hml3d_foot_contacts_r = 259, 263

kit_root_joint_l, kit_root_joint_r = 0, 4
kit_local_joint_pos_l, kit_local_joint_pos_r = 4, 64  # 4, 4+20*3
kit_joint_rotation_l, kit_joint_rotation_r = 64, 184  # 4+20*3, 4+20*3+20*6
kit_joint_velocity_l, kit_joint_velocity_r = 184, 247  # 4+20*3+20*6+21*3, -4
kit_foot_contacts_l, kit_foot_contacts_r = 247, 251

def hml3d_to_gcn_format(raw_data, joint_format):
    
    is_batch_data = len(raw_data.shape) == 3
    L, D = raw_data.shape[-2:]
    if is_batch_data:
        data = raw_data.reshape([-1, D])
    else:
        data = raw_data
    if joint_format == "hml3d_joint_1":
        njoints = 22

        assert data.shape[1] == 263
        new_data = np.zeros([data.shape[0], njoints + 2, 12])  # 12 = 3*2+6
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        # root point
        new_data[:, 0, :4] = data[:, hml3d_root_joint_l:hml3d_root_joint_r]
        new_data[:, 0, 4:7] = data[:, hml3d_joint_velocity_l:hml3d_joint_velocity_l + 3]
        # other skeleton joints
        new_data[:, 1:njoints, :3] = data[:, hml3d_local_joint_pos_l:hml3d_local_joint_pos_r].reshape(-1, njoints - 1, 3)
        new_data[:, 1:njoints, 3:9] = data[:, hml3d_joint_rotation_l:hml3d_joint_rotation_r].reshape(-1, njoints - 1, 6)
        new_data[:, 1:njoints, 9:] = data[:, hml3d_joint_velocity_l + 3:hml3d_joint_velocity_r].reshape(-1, njoints - 1, 3)
        # extract joint for foot contacts
        new_data[:, -2, :4] = data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r]
        
    elif joint_format == "kit_joint_1":
        njoints = 21
        assert data.shape[1] == 251
        new_data = np.zeros([data.shape[0], njoints + 1, 12])  # 12 = 3*2+6
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        # root point
        new_data[:, 0, :4] = data[:, kit_root_joint_l:kit_root_joint_r]
        new_data[:, 0, 4:7] = data[:, kit_joint_velocity_l:kit_joint_velocity_l + 3]
        # other skeleton joints
        new_data[:, 1:njoints, :3] = data[:, kit_local_joint_pos_l:kit_local_joint_pos_r].reshape(-1, njoints - 1,
                                                                                                      3)
        new_data[:, 1:njoints, 3:9] = data[:, kit_joint_rotation_l:kit_joint_rotation_r].reshape(-1, njoints - 1, 6)
        new_data[:, 1:njoints, 9:] = data[:, kit_joint_velocity_l + 3:kit_joint_velocity_r].reshape(-1, njoints - 1,
                                                                                                        3)
        # extract joint for foot contacts
        new_data[:, -1, :4] = data[:, kit_foot_contacts_l:kit_foot_contacts_r]
    elif joint_format == "behave_joint_1":
        njoints = 22
        assert data.shape[1] == 269
        new_data = np.zeros([data.shape[0], njoints + 2, 12])  # 12 = 3*2+6(0:3=xyz)（3:9=rotate）(9:12=速度)（12：y-angle）（13:16=trans）
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        # root point
        new_data[:, 0, :4] = data[:, hml3d_root_joint_l:hml3d_root_joint_r]
        new_data[:, 0, 4:7] = data[:, hml3d_joint_velocity_l:hml3d_joint_velocity_l + 3]
        # other skeleton joints
        new_data[:, 1:njoints, :3] = data[:, hml3d_local_joint_pos_l:hml3d_local_joint_pos_r].reshape(-1, njoints - 1, 3)
        new_data[:, 1:njoints, 3:9] = data[:, hml3d_joint_rotation_l:hml3d_joint_rotation_r].reshape(-1, njoints - 1, 6)
        new_data[:, 1:njoints, 9:] = data[:, hml3d_joint_velocity_l + 3:hml3d_joint_velocity_r].reshape(-1, njoints - 1, 3)
        # extract joint for foot contacts
        new_data[:, -2, :4] = data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r]
        # object
        new_data[:,-1,0:6]=data[:, 263:]
    
    elif joint_format == "behave_joint_2":
        njoints = 22
        assert data.shape[1] == 295
        new_data = np.zeros([data.shape[0], njoints + 2, 13])  # 12 = 3*2+6(0:3=xyz)（3:9=rotate）(9:12=速度)（12：y-angle）（13:16=trans）
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        # root point
        new_data[:, 0, :4] = data[:, hml3d_root_joint_l:hml3d_root_joint_r]
        new_data[:, 0, 4:7] = data[:, hml3d_joint_velocity_l:hml3d_joint_velocity_l + 3]
        #6D
        new_data[:,0,7:11]=data[:,269:273]
        # other skeleton joints
        new_data[:, 1:njoints, :3] = data[:, hml3d_local_joint_pos_l:hml3d_local_joint_pos_r].reshape(-1, njoints - 1, 3)
        new_data[:, 1:njoints, 3:9] = data[:, hml3d_joint_rotation_l:hml3d_joint_rotation_r].reshape(-1, njoints - 1, 6)
        new_data[:, 1:njoints, 9:12] = data[:, hml3d_joint_velocity_l + 3:hml3d_joint_velocity_r].reshape(-1, njoints - 1, 3)
        
        # extract joint for foot contacts
        new_data[:, -2, :4] = data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r]
        # object_motion
        new_data[:,-1,3:9]=data[:, 263:269]
        #concat
        new_data[:, 0:njoints, 12]=data[:,273:].reshape(-1, njoints)
    elif joint_format == "behave_joint_3":
        njoints = 22
        assert data.shape[1] == 291
        new_data = np.zeros([data.shape[0], njoints + 2, 13])  # 12 = 3*2+6(0:3=xyz)（3:9=rotate）(9:12=速度)（12：y-angle）（13:16=trans）
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        # root point
        new_data[:, 0, :4] = data[:, hml3d_root_joint_l:hml3d_root_joint_r]
        new_data[:, 0, 4:7] = data[:, hml3d_joint_velocity_l:hml3d_joint_velocity_l + 3]
        # other skeleton joints
        new_data[:, 1:njoints, :3] = data[:, hml3d_local_joint_pos_l:hml3d_local_joint_pos_r].reshape(-1, njoints - 1, 3)
        new_data[:, 1:njoints, 3:9] = data[:, hml3d_joint_rotation_l:hml3d_joint_rotation_r].reshape(-1, njoints - 1, 6)
        new_data[:, 1:njoints, 9:12] = data[:, hml3d_joint_velocity_l + 3:hml3d_joint_velocity_r].reshape(-1, njoints - 1, 3)
        
        # extract joint for foot contacts
        new_data[:, -2, :4] = data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r]
        # object
        new_data[:,-1,3:9]=data[:, 263:269]
        #concat
        new_data[:, 0:njoints, 12]=data[:,269:].reshape(-1, njoints)
    elif joint_format == "behave_joint_4":
        njoints = 22
        assert data.shape[1] == 295
        new_data = np.zeros([data.shape[0], njoints + 1, 14])  # 12 = 3*2+6(0:3=xyz)（3:9=rotate）(9:12=速度)（12：y-angle）（13:16=trans）
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        # root point
        new_data[:, 0, :4] = data[:, hml3d_root_joint_l:hml3d_root_joint_r]
        new_data[:, 0, 4:7] = data[:, hml3d_joint_velocity_l:hml3d_joint_velocity_l + 3]
        #6D
        new_data[:,0,7:11]=data[:,269:273]
        # other skeleton joints
        new_data[:, 1:njoints, :3] = data[:, hml3d_local_joint_pos_l:hml3d_local_joint_pos_r].reshape(-1, njoints - 1, 3)
        new_data[:, 1:njoints, 3:9] = data[:, hml3d_joint_rotation_l:hml3d_joint_rotation_r].reshape(-1, njoints - 1, 6)
        new_data[:, 1:njoints, 9:12] = data[:, hml3d_joint_velocity_l + 3:hml3d_joint_velocity_r].reshape(-1, njoints - 1, 3)
        
        # extract joint for foot contacts
        new_data[:, 7, 12:13] = data[:, hml3d_joint_velocity_l :hml3d_joint_velocity_l+1]
        new_data[:, 8, 12:13] = data[:, hml3d_joint_velocity_l+1 :hml3d_joint_velocity_l+2]
        new_data[:, 10, 12:13] = data[:, hml3d_joint_velocity_l+2 :hml3d_joint_velocity_l+3]
        new_data[:, 11, 12:13] = data[:, hml3d_joint_velocity_l+3 :hml3d_joint_velocity_l+4]
        #new_data[:, -2, :4] = data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r]
        # object_motion
        new_data[:,-1,0:6]=data[:, 263:269]
        #concat
        new_data[:, 0:njoints, 13]=data[:,273:].reshape(-1, njoints)
    elif joint_format == "behave_joint_5":
        njoints = 22
        assert data.shape[1] == 273
        new_data = np.zeros([data.shape[0], njoints + 2, 12])  # 12 = 3*2+6(0:3=xyz)（3:9=rotate）(9:12=速度)（12：y-angle）（13:16=trans）
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        # root point
        new_data[:, 0, :4] = data[:, hml3d_root_joint_l:hml3d_root_joint_r]
        new_data[:, 0, 4:7] = data[:, hml3d_joint_velocity_l:hml3d_joint_velocity_l + 3]
        #6D
        new_data[:,0,7:11]=data[:,269:273]
        # other skeleton joints
        new_data[:, 1:njoints, :3] = data[:, hml3d_local_joint_pos_l:hml3d_local_joint_pos_r].reshape(-1, njoints - 1, 3)
        new_data[:, 1:njoints, 3:9] = data[:, hml3d_joint_rotation_l:hml3d_joint_rotation_r].reshape(-1, njoints - 1, 6)
        new_data[:, 1:njoints, 9:12] = data[:, hml3d_joint_velocity_l + 3:hml3d_joint_velocity_r].reshape(-1, njoints - 1, 3)
        
        # extract joint for foot contacts
        new_data[:, -2, :4] = data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r]
        #new_data[:, -2, 8:] = -1
        # object_motion
        new_data[:,-1,0:6]=data[:, 263:269]
    elif joint_format == "behave_joint_6":
        
        njoints = 22
        assert data.shape[1] == 291
        new_data = np.zeros([data.shape[0], njoints + 3, 22])  # 12 = 3*2+6(0:3=xyz)（3:9=rotate）(9:12=速度)（12：y-angle）（13:16=trans）
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        # root point
        new_data[:, 0, :4] = data[:, hml3d_root_joint_l:hml3d_root_joint_r]
        new_data[:, 0, 4:7] = data[:, hml3d_joint_velocity_l:hml3d_joint_velocity_l + 3]
        # other skeleton joints
        new_data[:, 1:njoints, :3] = data[:, hml3d_local_joint_pos_l:hml3d_local_joint_pos_r].reshape(-1, njoints - 1, 3)
        new_data[:, 1:njoints, 3:9] = data[:, hml3d_joint_rotation_l:hml3d_joint_rotation_r].reshape(-1, njoints - 1, 6)
        new_data[:, 1:njoints, 9:12] = data[:, hml3d_joint_velocity_l + 3:hml3d_joint_velocity_r].reshape(-1, njoints - 1, 3)
        
        # extract joint for foot contacts
        new_data[:, -3, :4] = data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r]
        # extract contact for contacts
        new_data[:, -2, :22] = data[:,269:]
        # object_motion
        new_data[:,-1,0:6]=data[:, 263:269]

    elif joint_format == "behave_joint_7":
        njoints = 22
        assert data.shape[1] == 269
        new_data = np.zeros([data.shape[0], njoints + 2, 12])  # 12 = 3*2+6(0:3=xyz)（3:9=rotate）(9:12=速度)（12：y-angle）（13:16=trans）
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        # root point
        new_data[:, 0, :4] = data[:, hml3d_root_joint_l:hml3d_root_joint_r]
        new_data[:, 0, 4:7] = data[:, hml3d_joint_velocity_l:hml3d_joint_velocity_l + 3]
        # other skeleton joints
        new_data[:, 1:njoints, :3] = data[:, hml3d_local_joint_pos_l:hml3d_local_joint_pos_r].reshape(-1, njoints - 1, 3)
        new_data[:, 1:njoints, 3:9] = data[:, hml3d_joint_rotation_l:hml3d_joint_rotation_r].reshape(-1, njoints - 1, 6)
        new_data[:, 1:njoints, 9:12] = data[:, hml3d_joint_velocity_l + 3:hml3d_joint_velocity_r].reshape(-1, njoints - 1, 3)
        
        # extract joint for foot contacts
        new_data[:, -2, :4] = data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r]
        # object_motion
        new_data[:,-1,0:6]=data[:, 263:269]

    else:
        raise ValueError(f"joint format <{joint_format}> not implemented.")
    
    if is_batch_data:

        new_data = new_data.reshape([-1, L] + [new_data.shape[-2],new_data.shape[-1]])

    return new_data


def gcn_to_hml3d_format(raw_data, joint_format):
    is_batch_data = len(raw_data.shape) == 4
    L, J, D = raw_data.shape[-3:]
    if is_batch_data:
        data = raw_data.reshape([-1, J, D])
    else:
        data = raw_data

    if joint_format == "hml3d_joint_1":
        njoints = 22
        assert data.shape[1] == njoints + 1
        new_data = np.zeros([data.shape[0], 263])
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        new_data[:, hml3d_root_joint_l:hml3d_root_joint_r] = data[:, 0, :4]
        new_data[:, hml3d_local_joint_pos_l: hml3d_local_joint_pos_r] = data[:, 1:njoints, :3].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_joint_rotation_l: hml3d_joint_rotation_r] = data[:, 1:njoints, 3:9].reshape(-1, (njoints - 1) * 6)
        new_data[:, hml3d_joint_velocity_l: hml3d_joint_velocity_l + 3] = data[:, 0, 4:7]
        new_data[:, hml3d_joint_velocity_l + 3: hml3d_joint_velocity_r] = data[:, 1:njoints, 9:].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r] = data[:, -1, :4]
    elif joint_format == "kit_joint_1":
        njoints = 21
        assert data.shape[1] == njoints + 1
        new_data = np.zeros([data.shape[0], 251])
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        new_data[:, kit_root_joint_l:kit_root_joint_r] = data[:, 0, :4]
        new_data[:, kit_local_joint_pos_l: kit_local_joint_pos_r] = data[:, 1:njoints, :3].reshape(-1, (
                    njoints - 1) * 3)
        new_data[:, kit_joint_rotation_l: kit_joint_rotation_r] = data[:, 1:njoints, 3:9].reshape(-1,
                                                                                                      (njoints - 1) * 6)
        new_data[:, kit_joint_velocity_l: kit_joint_velocity_l + 3] = data[:, 0, 4:7]
        new_data[:, kit_joint_velocity_l + 3: kit_joint_velocity_r] = data[:, 1:njoints, 9:].reshape(-1, (
                    njoints - 1) * 3)
        new_data[:, kit_foot_contacts_l:kit_foot_contacts_r] = data[:, -1, :4]
    elif joint_format == "behave_joint_1":
        njoints = 22
        
        assert data.shape[1] == njoints + 2
        new_data = np.zeros([data.shape[0], 269])
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        new_data[:, hml3d_root_joint_l:hml3d_root_joint_r] = data[:, 0, :4]
        new_data[:, hml3d_local_joint_pos_l: hml3d_local_joint_pos_r] = data[:, 1:njoints, :3].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_joint_rotation_l: hml3d_joint_rotation_r] = data[:, 1:njoints, 3:9].reshape(-1, (njoints - 1) * 6)
        new_data[:, hml3d_joint_velocity_l: hml3d_joint_velocity_l + 3] = data[:, 0, 4:7]
        new_data[:, hml3d_joint_velocity_l + 3: hml3d_joint_velocity_r] = data[:, 1:njoints, 9:].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r] = data[:, -2, :4]
        new_data[:, 263:] = data[:, -1, 0:6]
    elif joint_format == "behave_joint_2":
        njoints = 22
        assert data.shape[1] == njoints + 2
        new_data = np.zeros([data.shape[0], 295])
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        new_data[:, hml3d_root_joint_l:hml3d_root_joint_r] = data[:, 0, :4]
        new_data[:, hml3d_local_joint_pos_l: hml3d_local_joint_pos_r] = data[:, 1:njoints, :3].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_joint_rotation_l: hml3d_joint_rotation_r] = data[:, 1:njoints, 3:9].reshape(-1, (njoints - 1) * 6)
        new_data[:, hml3d_joint_velocity_l: hml3d_joint_velocity_l + 3] = data[:, 0, 4:7]
        new_data[:, hml3d_joint_velocity_l + 3: hml3d_joint_velocity_r] = data[:, 1:njoints, 9:12].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r] = data[:, -2, :4]
        new_data[:, 263:269] = data[:, -1, 3:9]
        new_data[:, 269:273] = data[:, 0, 7:11]
        new_data[:, 273:] = data[:, 0:njoints, 12].reshape(-1,njoints)
    elif joint_format == "behave_joint_3":
        njoints = 22
        assert data.shape[1] == njoints + 2
        new_data = np.zeros([data.shape[0], 291])
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        new_data[:, hml3d_root_joint_l:hml3d_root_joint_r] = data[:, 0, :4]
        new_data[:, hml3d_local_joint_pos_l: hml3d_local_joint_pos_r] = data[:, 1:njoints, :3].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_joint_rotation_l: hml3d_joint_rotation_r] = data[:, 1:njoints, 3:9].reshape(-1, (njoints - 1) * 6)
        new_data[:, hml3d_joint_velocity_l: hml3d_joint_velocity_l + 3] = data[:, 0, 4:7]
        new_data[:, hml3d_joint_velocity_l + 3: hml3d_joint_velocity_r] = data[:, 1:njoints, 9:12].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r] = data[:, -2, :4]
        new_data[:, 263:269] = data[:, -1, 3:9]
        new_data[:, 269:] = data[:, 0:njoints, 12].reshape(-1,njoints)
    elif joint_format == "behave_joint_4":
        njoints = 22
        assert data.shape[1] == njoints + 1
        new_data = np.zeros([data.shape[0], 295])
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        new_data[:, hml3d_root_joint_l:hml3d_root_joint_r] = data[:, 0, :4]
        new_data[:, hml3d_local_joint_pos_l: hml3d_local_joint_pos_r] = data[:, 1:njoints, :3].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_joint_rotation_l: hml3d_joint_rotation_r] = data[:, 1:njoints, 3:9].reshape(-1, (njoints - 1) * 6)
        new_data[:, hml3d_joint_velocity_l: hml3d_joint_velocity_l + 3] = data[:, 0, 4:7]
        new_data[:, hml3d_joint_velocity_l + 3: hml3d_joint_velocity_r] = data[:, 1:njoints, 9:12].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_l+1] = data[:, 7, 12:13]
        new_data[:, hml3d_foot_contacts_l+1:hml3d_foot_contacts_l+2] = data[:, 8, 12:13]
        new_data[:, hml3d_foot_contacts_l+2:hml3d_foot_contacts_l+3] = data[:, 10, 12:13]
        new_data[:, hml3d_foot_contacts_l+3:hml3d_foot_contacts_l+4] = data[:, 11, 12:13]
        new_data[:, 263:269] = data[:, -1, 0:6]
        new_data[:, 269:273] = data[:, 0, 7:11]
        new_data[:, 273:] = data[:, 0:njoints, 13].reshape(-1,njoints)
    elif joint_format == "behave_joint_5":
        njoints = 22
        assert data.shape[1] == njoints + 2
        new_data = np.zeros([data.shape[0], 273])
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        new_data[:, hml3d_root_joint_l:hml3d_root_joint_r] = data[:, 0, :4]
        new_data[:, hml3d_local_joint_pos_l: hml3d_local_joint_pos_r] = data[:, 1:njoints, :3].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_joint_rotation_l: hml3d_joint_rotation_r] = data[:, 1:njoints, 3:9].reshape(-1, (njoints - 1) * 6)
        new_data[:, hml3d_joint_velocity_l: hml3d_joint_velocity_l + 3] = data[:, 0, 4:7]
        new_data[:, hml3d_joint_velocity_l + 3: hml3d_joint_velocity_r] = data[:, 1:njoints, 9:12].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r] = data[:, -2, :4]
        new_data[:, 263:269] = data[:, -1, 0:6]
        new_data[:, 269:273] = data[:, 0, 7:11]
    elif joint_format == "behave_joint_6":
        njoints = 22
        assert data.shape[1] == njoints + 3
        new_data = np.zeros([data.shape[0], 291])
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
        new_data[:, hml3d_root_joint_l:hml3d_root_joint_r] = data[:, 0, :4]
        new_data[:, hml3d_local_joint_pos_l: hml3d_local_joint_pos_r] = data[:, 1:njoints, :3].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_joint_rotation_l: hml3d_joint_rotation_r] = data[:, 1:njoints, 3:9].reshape(-1, (njoints - 1) * 6)
        new_data[:, hml3d_joint_velocity_l: hml3d_joint_velocity_l + 3] = data[:, 0, 4:7]
        new_data[:, hml3d_joint_velocity_l + 3: hml3d_joint_velocity_r] = data[:, 1:njoints, 9:12].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r] = data[:, -3, :4]
        new_data[:, 269:] = data[:, -2, 0:22]
        new_data[:, 263:269] = data[:, -1, 0:6]
    elif joint_format == "behave_joint_7":

        njoints = 22
        assert data.shape[1] == njoints + 2
        new_data = np.zeros([data.shape[0], 269])
        if type(data) == torch.Tensor:
            new_data = torch.from_numpy(new_data).to(data.device)
            new_data = new_data.type(data.dtype)
        new_data[:, hml3d_root_joint_l:hml3d_root_joint_r] = data[:, 0, :4]
        new_data[:, hml3d_local_joint_pos_l: hml3d_local_joint_pos_r] = data[:, 1:njoints, :3].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_joint_rotation_l: hml3d_joint_rotation_r] = data[:, 1:njoints, 3:9].reshape(-1, (njoints - 1) * 6)
        new_data[:, hml3d_joint_velocity_l: hml3d_joint_velocity_l + 3] = data[:, 0, 4:7]
        new_data[:, hml3d_joint_velocity_l + 3: hml3d_joint_velocity_r] = data[:, 1:njoints, 9:12].reshape(-1, (njoints - 1) * 3)
        new_data[:, hml3d_foot_contacts_l:hml3d_foot_contacts_r] = data[:, -2, :4]
        new_data[:, 263:269] = data[:, -1, 0:6]
    else:
        raise ValueError(f"joint format <{joint_format}> not implemented.")

    if is_batch_data:
        new_data = new_data.reshape([-1, L, new_data.shape[-1]])

    return new_data


