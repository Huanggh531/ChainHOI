import os
import os.path as osp
import shutil

import imageio
import matplotlib

import numpy as np
import torch

import bpy
import matplotlib.pyplot as plt
from utils.rotation2xyz import Rotation2xyz
from utils.simplify_loc2rot import joints2smpl

from blender.scene import setup_scene
from blender.tools import load_numpy_vertices_into_blender, delete_objs
from blender.materials import body_material
from blender.floor import plot_floor


def prune_begin_end(data, perc):
    to_remove = int(len(data) * perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]

import math



class Camera:
    def __init__(self, first_root, mode):
        camera = bpy.data.objects['Camera']

        ## initial position
        # camera.location.x = -5

        # camera.location.y = -10 
        # camera.location.z = 0

        camera.location.x = 7.36 + 0.5
        camera.location.y = -6.93
        camera.location.z = 5.6

        # wider point of view
        if mode == "sequence":
            camera.data.lens = 80
        elif mode == "frame":
            camera.data.lens = 130
        elif mode == "video":
            camera.data.lens = 110

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]
        # camera.rotation_euler[0] = math.radians(95)
        # camera.rotation_euler[1] = math.radians(0)
        # camera.rotation_euler[2] = math.radians(-30)

        self._root = first_root

    def update(self, newroot):
        delta_root = newroot - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]

        self._root = newroot


class Camera2:
    def __init__(self, first_root, mode):
        camera = bpy.data.objects['Camera']

        ## initial position
        # camera.location = [12, 0,  1.2]
        camera.location = [11, 0,  1.2]
        camera.rotation_mode = 'XYZ'
        camera.rotation_euler = (np.pi/2, 0, np.pi/2)
 
        camera.data.lens = 110

        self.mode = mode
        self.camera = camera
        self.first_root = first_root

    def update(self, newroot):
        pass
        # dis = newroot - self.first_root
        # self.camera.location = np.array([12, 0, 1.2]) + dis


def blender_render(motions, faces, name, frame_dir, video_dir, sequence_dir,
                   mode="video", vis_frame_num=8, color="Blues", on_floor=True,
                   down_sample=8, disable_floor=False, keyframe_id=[], only_keyframe=False,
                   image_quality="med"):
    setup_scene(res=image_quality,
                denoising=True,
                oldrender=True,
                accelerator="gpu",
                device=[0])

    motions = motions[..., [2, 0, 1]]
    if on_floor:
        #motions[..., 2] -= motions[..., 2].min(1)[:, None]
        motions[:,:6890, 2] -= motions[:,:6890,2].min(1)[:, None]
        #motions[:,6890:, 2] -= motions[:,6890:,2].min(1)[:, None]
    else:
        #motions[..., 2] -= motions[..., 2].min()
        motions[:,:6890, 2] -= motions[:,:6890, 2].min()
        #motions[:,6890:, 2] -= motions[:,6890:, 2].min()
    
    # motions[..., 2] *= -1
    # # if on_floor:
    # #     #motions[..., 2] -= motions[..., 2].min(1)[:, None]
    # #     motions[:,:6890, 2] += motions[:,:6890,2].min(1)[:, None]
    # #     motions[:,6890:, 2] += motions[:,6890:,2].min(1)[:, None]
    # # else:
    # #     #motions[..., 2] -= motions[..., 2].min()
    # #     motions[:,:6890, 2] += motions[:,:6890, 2].min()
    # #     motions[:,6890:, 2] += motions[:,6890:, 2].min()

    if disable_floor is not True:
        plot_floor(motions, big_plane=False)
    if mode == "frame":
        camera = Camera2(first_root=motions[0].mean(0), mode=mode)
    else:
        camera = Camera(first_root=motions[0].mean(0), mode=mode)
    

    cmap = plt.get_cmap(color)

    imported_obj_names = []
    if mode == "sequence":
        # motions = prune_begin_end(motions, perc=0.1)
        frame_idx = list(np.round(np.linspace(0, len(motions) - 1, vis_frame_num)).astype(int))
        color_begin, color_end = 0.5, 0.9
        camera.update(motions.mean((0, 1)))

        for _, idx in enumerate(frame_idx):
            r = _ / (vis_frame_num - 1)
            cur_body_color = cmap(color_begin + (color_end - color_begin) * r)
            cur_body_meterial = body_material(*cur_body_color, oldrender=True)
            load_numpy_vertices_into_blender(motions[idx], faces, "%03d" % idx, cur_body_meterial)
            imported_obj_names.append("%03d" % idx)

        bpy.context.scene.render.filepath = os.path.join(os.getcwd(), os.path.join(sequence_dir, f"{name}.png"))
        bpy.ops.render.render(use_viewport=True, write_still=True)
    elif mode == "frame":
        frame_dir = os.path.join(frame_dir, name)
        os.makedirs(frame_dir, exist_ok=True)

        # cur_body_meterial = body_material(0.035, 0.322, 0.615)
        cur_body_meterial = body_material(255/255, 165/255, 0/255)

        for i in range(len(motions)):
            if i % down_sample != 0:
                continue
            camera.update(motions[i].mean(0))
            load_numpy_vertices_into_blender(motions[i], faces, "%03d" % i, cur_body_meterial)

            bpy.context.scene.render.filepath = os.path.join(os.getcwd(), os.path.join(frame_dir, f"frame_{i:03}.png"))
            bpy.ops.render.render(use_viewport=True, write_still=True)
            delete_objs("%03d" % i)

    elif mode == "video":
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(osp.join(video_dir, "test_"+name), exist_ok=True)

        cur_body_meterial = body_material(0.035, 0.322, 0.615)

        for i in range(len(motions)):
            camera.update(motions[i].mean(0))
            load_numpy_vertices_into_blender(motions[i], faces, "%03d" % i, cur_body_meterial)

            bpy.context.scene.render.filepath = os.path.join(os.getcwd(), os.path.join(video_dir, "test_"+name, f"frame_{i:03}.png"))
            bpy.ops.render.render(use_viewport=True, write_still=True)
            delete_objs("%03d" % i)

        frames = []

        for i in range(len(motions)):
            frames.append(imageio.imread(osp.join(video_dir, "test_"+name, f"frame_{i:03}.png")))

        out = np.stack(frames, axis=0)
        imageio.mimsave(osp.join(video_dir, name + '.mp4'), out, fps=20)
        shutil.rmtree(osp.join(video_dir, "test_"+name))
    elif mode == 'keyframe':
        frame_idx = list(np.round(np.linspace(0, len(motions) - 1, vis_frame_num)).astype(int))
        color_begin, color_end = 0.5, 0.9
        camera.update(motions.mean((0, 1)))

        for _, idx in enumerate(frame_idx):
            if _ in keyframe_id:
                cur_body_meterial = body_material(255/255, 165/255, 0/255)
            else:
                r = _ / (vis_frame_num - 1)
                cur_body_color = cmap(color_begin + (color_end - color_begin) * r)
                cur_body_meterial = body_material(*cur_body_color, oldrender=True)
            if not (only_keyframe and _ not in keyframe_id):
                load_numpy_vertices_into_blender(motions[idx], faces, "%03d" % idx, cur_body_meterial)
                imported_obj_names.append("%03d" % idx)

        bpy.context.scene.render.filepath = os.path.join(os.getcwd(), os.path.join(sequence_dir, f"{name}.png"))
        bpy.ops.render.render(use_viewport=True, write_still=True)
    else:
        raise ValueError(f"{mode} not supported!")

    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])


def get_motion_meshes(data, name, device, mesh_dir):
    motions=data["human"]
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset

    rot2xyz = Rotation2xyz(device=device)
    faces = rot2xyz.smpl_model.faces
    
    if not os.path.exists(osp.join(mesh_dir, name + '.pt')):
        if device == "cpu":
            j2s = joints2smpl(num_frames=frames, device_id=0, cuda=False)
        else:
            j2s = joints2smpl(num_frames=frames, device_id=device.index, cuda=True)

        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3] for hml3d

        vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                           pose_rep='rot6d', translation=True, glob=True,
                           jointstype='vertices',
                           vertstrans=True).cpu()

        #torch.save(vertices.cpu(), osp.join(mesh_dir, name + '.pt'))
        vertices = vertices[0].permute([2, 0, 1]).numpy()
        # 合并顶点
        vertices_combined = np.concatenate((vertices, data["obj"]), axis=1)
        # 合并面，并调整faces2的索引
        faces_combined = np.concatenate((faces, data["faces"] + vertices.shape[1]),axis=0)

        np.save("result/vertices",vertices_combined)
        np.save("result/faces",faces_combined)
    else:
        vertices_combined = torch.load(osp.join(mesh_dir, name + '.pt'))
        faces_combined = vertices[0].permute([2, 0, 1]).numpy()
        
    return vertices_combined, faces_combined


from scipy.ndimage import gaussian_filter
def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

def get_motion_meshes2(data, name, device, mesh_dir):
    
    motions=data["human"]
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset

    motions = motion_temporal_filter(motions, sigma=2.5)
    data["obj"] = motion_temporal_filter(data["obj"], sigma=2.5)
    rot2xyz = Rotation2xyz(device=device)
    faces = rot2xyz.smpl_model.faces

    # if (not os.path.exists(osp.join(mesh_dir, name))) or (len(os.listdir(osp.join(mesh_dir, name)))==0):
    if device == "cpu":
        j2s = joints2smpl(num_frames=frames, device_id=0, cuda=False)
    else:
        j2s = joints2smpl(num_frames=frames, device_id=device.index, cuda=True)

    print(f'Running SMPLify, it may take a few minutes.')

    motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3] for hml3d

    vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                       pose_rep='rot6d', translation=True, glob=True,
                       jointstype='vertices',
                       vertstrans=True).cpu()

    #torch.save(vertices.cpu(), osp.join(mesh_dir, name + '.pt'))
    vertices = vertices[0].permute([2, 0, 1]).numpy()
    # 合并顶点
    vertices_combined = np.concatenate((vertices, data["obj"]), axis=1)
    # 合并面，并调整faces2的索引
    faces_combined = np.concatenate((faces, data["faces"] + vertices.shape[1]),axis=0)
    if not os.path.exists(osp.join(mesh_dir, name)):
        os.makedirs(osp.join(mesh_dir, name))
    np.save(osp.join(mesh_dir, name,"human_vertices"),vertices)
    np.save(osp.join(mesh_dir, name,"human_faces"),faces)
    np.save(osp.join(mesh_dir, name,"obj_vertices"),data["obj"])
    np.save(osp.join(mesh_dir, name,"obj_faces"),data["faces"])

    np.save(osp.join(mesh_dir, name,"vertices"),vertices_combined)
    np.save(osp.join(mesh_dir, name,"faces"),faces_combined)
    # else:
    #     vertices_combined = np.load(osp.join(mesh_dir, name,"vertices.npy"))
    #     faces_combined = np.load(osp.join(mesh_dir, name,"faces.npy"))
        
    
    
    return vertices_combined, faces_combined


def render_image(motions, name, frame_dir, video_dir, sequence_dir, mesh_dir,
                 mode, vis_frame_num, color, device, on_floor, down_sample, disable_floor,
                 keyframe_id, only_keyframe, image_quality):

    vertices, faces = get_motion_meshes2(motions, name, device, mesh_dir)
    #vertices=np.load("/home/guohong/HOI/result/vertices.npy")
    #faces=np.load("/home/guohong/HOI/result/faces.npy")
    # blender_render(vertices, faces, name, frame_dir, video_dir, sequence_dir,
    #                mode=mode, vis_frame_num=vis_frame_num, color=color, on_floor=on_floor,
    #                down_sample=down_sample, disable_floor=disable_floor, keyframe_id=keyframe_id,
    #                only_keyframe=only_keyframe, image_quality=image_quality)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="./visual_datas/motion_hml3d", help='motion npy file dir',
                        required=False)
    parser.add_argument("--frame_dir", type=str, default="./visual_datas/frames/", help='save dir')
    parser.add_argument("--video_dir", type=str, default="./visual_datas/omomo/", help='save dir')
    parser.add_argument("--sequence_dir", type=str, default="./visual_datas/sequences/", help='save dir')
    parser.add_argument("--mesh_dir", type=str, default="./visual_datas/omomo/", help='save dir')
    #parser.add_argument('--motion_list', default="002103", nargs="+", type=str, help="motion name list")
    parser.add_argument('--motion_path', default="", type=str)
    parser.add_argument("--mode", type=str, default="sequence")
    parser.add_argument("--image_quality", type=str, default="med")
    # for mode=sequence
    parser.add_argument("--vis_frame_num", type=int, default=8)
    # for mode=frame
    parser.add_argument("--down_sample", type=int, default=8)
    # for mode=keyframe
    parser.add_argument("--keyframe_id", default=-1, nargs="+", type=int)
    parser.add_argument("--only_keyframe", action="store_true", default=False)

    parser.add_argument("--color", type=str, default="Blues")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--on_floor", action="store_true", default=False)
    parser.add_argument("--disable_floor", action="store_true", default=False)
    return parser.parse_args()


def go():
    args = get_args()

    device = args.device
    if device != "cpu":
        device = torch.device(f"cuda:{device}")

    # print(os.listdir(os.getcwd()))
    # exit()

    #filename_list = args.motion_list
    filename_path = args.motion_path
    #filename_list=filename_list[0]
    filename_list=os.listdir(filename_path)
    file_dir = args.file_dir

    for index,filename in enumerate(filename_list):
        
        #sub_files=os.listdir(os.path.join(filename_path,filename))

        # for file in sub_files:
        data={}
        #     import pdb
        #     pdb.set_trace()
        #     #name=filename+"_"+file
        npy_files=os.listdir(os.path.join(filename_path,filename))
        import pdb
        pdb.set_trace()
        for index,npy in enumerate(npy_files):
            name = npy.split(".")[0]
            if index==3:
                break
            data[name]=np.load(os.path.join(filename_path,filename,npy))

        print('processing: ', filename+"_")
        render_image(data, name=filename+"_", frame_dir=args.frame_dir, video_dir=args.video_dir,
                  sequence_dir=args.sequence_dir, mesh_dir=args.mesh_dir, mode=args.mode,
                  vis_frame_num=args.vis_frame_num, color=args.color, device=device,
                  on_floor=args.on_floor, down_sample=args.down_sample, disable_floor=args.disable_floor,
                  keyframe_id=args.keyframe_id, only_keyframe=args.only_keyframe, image_quality=args.image_quality)
        print("-------{}/{}--------".format(index,len(filename_list)))
        # import pdb
        # pdb.set_trace()
    # for filename in filename_list:
    #     filename=filename.split(".")[0]
    #     motion = np.load(osp.join(file_dir, filename + ".npy"))
    #     print('processing: ', filename, motion.shape)
    #     render_image(motion, name=filename, frame_dir=args.frame_dir, video_dir=args.video_dir,
    #                  sequence_dir=args.sequence_dir, mesh_dir=args.mesh_dir, mode=args.mode,
    #                  vis_frame_num=args.vis_frame_num, color=args.color, device=device,
    #                  on_floor=args.on_floor, down_sample=args.down_sample, disable_floor=args.disable_floor,
    #                  keyframe_id=args.keyframe_id, only_keyframe=args.only_keyframe, image_quality=args.image_quality)


if __name__ == "__main__":
    go()

#CUDA_VISIBLE_DEVICES=2 python visualize/blend_render.py --motion_path /home/guohong/HOI/results/pred_motion_roberta_obj_no --mode "video" --video_dir "./visual_datas/pred_motion_roberta_obj_yes"
