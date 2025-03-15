import os
import os.path as osp
import shutil
import time

import imageio
import matplotlib

import numpy as np
import torch

import bpy

from utils.mld_simplify_loc2rot import joints2smpl

from blender.scene import setup_scene
from blender.tools import load_numpy_vertices_into_blender, delete_objs
from blender.materials import body_material
from blender.floor import plot_floor
from blender.meshes import Meshes

import sys

from scipy.ndimage import gaussian_filter


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)


def prune_begin_end(data, perc):
    to_remove = int(len(data) * perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


class Camera:
    def __init__(self, first_root, mode):
        camera = bpy.data.objects['Camera']

        ## initial position
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
        # camera.location.x = 7.36 + 0.5
        # camera.location.y = -6.93
        # camera.location.z = 5.6
        camera.data.lens = 90
        self.location = np.array([7.36 + 0.5, -6.93, 5.6])
        camera.location = self.location

        self.mode = mode
        self.camera = camera
        self.first_root = first_root

    def update(self, newroot):
        # pass
        dis = newroot - self.first_root
        self.camera.location = self.location + dis


def blender_render(meshes, name, frame_dir, video_dir, sequence_dir,
                   mode="video", vis_frame_num=8, color="Blues", on_floor=False,
                   down_sample=8, disable_floor=False, keyframe_id=[], only_keyframe=False,
                   image_quality="med"):
    print(meshes.shape)
    setup_scene(res=image_quality,
                denoising=True,
                oldrender=True,
                accelerator="gpu",
                device=[0])
    faces_path = "./visual_datas/render_deps/smplh/smplh.faces"

    data = Meshes(meshes,
                      gt=True,
                      mode=mode,
                      faces_path=faces_path,
                      canonicalize=True,
                      always_on_floor=False,
                      is_smplx=False)

    imported_obj_names = []

    if disable_floor is not True:
        plot_floor(data.data, big_plane=False)

    if mode == "frame":
        camera = Camera2(first_root=data.get_root(0), mode=mode)
    else:
        camera = Camera(first_root=data.get_root(0), mode=mode)

    frame_dir = os.path.join(frame_dir, name)
    if mode == "frame":
        os.makedirs(frame_dir, exist_ok=True)
        for i in range(len(data)):
            mat = data.mat
            if i % down_sample != 0:
                continue
            # camera.update(data.get_root(i))
            camera.update(data.data[i].mean(0))
            objname = data.load_in_blender(i, mat)
            bpy.context.scene.render.filepath = os.path.join(os.getcwd(), os.path.join(frame_dir, f"frame_{i:03}.png"))
            bpy.ops.render.render(use_viewport=True, write_still=True)
            delete_objs(objname)
    elif mode == "video":
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(osp.join(video_dir, "test_" + name), exist_ok=True)

        for i in range(len(data)):
            mat = data.mat
            # camera.update(data.get_root(i))
            camera.update(data.data[i].mean(0))
            objname = data.load_in_blender(i, mat)
            bpy.context.scene.render.filepath = os.path.join(os.getcwd(), os.path.join(frame_dir, f"frame_{i:03}.png"))
            bpy.ops.render.render(use_viewport=True, write_still=True)
            delete_objs(objname)

        frames = []
        for i in range(len(data)):
            frames.append(imageio.imread(osp.join(frame_dir, f"frame_{i:03}.png")))

        out = np.stack(frames, axis=0)
        imageio.mimsave(osp.join(video_dir, name + '.mp4'), out, fps=20)
        # shutil.rmtree(osp.join(video_dir, "test_" + name))
    else:
        print("mode error!")
        exit()

    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])


def get_motion_meshes(motions, name, device, mesh_dir):
    frames, njoints, nfeats = motions.shape

    motions = motion_temporal_filter(motions, sigma=2.5)

    j2s = joints2smpl(num_frames=frames, device_id=device.index, cuda=True)
    meshes = j2s.joint2smpl(motions)
    # meshes = j2s.joint2smpl_smooth(motions, min_cutoff=0.00001, beta=0.7)

    # if not os.path.exists(osp.join(mesh_dir, name + '.npy')):
    #     j2s = joints2smpl(num_frames=frames, device_id=device.index, cuda=True)
    #     # meshes = j2s.joint2smpl(motions)
    #     meshes = j2s.joint2smpl_smooth(motions, min_cutoff=0.004, beta=1.5)
    #     np.save(osp.join(mesh_dir, name + '.npy'), meshes)
    # else:
    #     meshes = np.load(osp.join(mesh_dir, name + '.npy'))

    return meshes


def render_image(motions, name, frame_dir, video_dir, sequence_dir, mesh_dir,
                 mode, vis_frame_num, color, device, on_floor, down_sample, disable_floor,
                 keyframe_id, only_keyframe, image_quality):
    vertices = get_motion_meshes(motions, name, device, mesh_dir)

    blender_render(vertices, name, frame_dir, video_dir, sequence_dir,
                   mode=mode, vis_frame_num=vis_frame_num, color=color, on_floor=on_floor,
                   down_sample=down_sample, disable_floor=disable_floor, keyframe_id=keyframe_id,
                   only_keyframe=only_keyframe, image_quality=image_quality)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="./visual_datas/motion_hml3d", help='motion npy file dir',
                        required=False)
    parser.add_argument("--frame_dir", type=str, default="./visual_datas/debug_samples/", help='save dir')
    parser.add_argument("--video_dir", type=str, default="./visual_datas/debug_videos/", help='save dir')
    parser.add_argument("--sequence_dir", type=str, default="./visual_datas/sequences/", help='save dir')
    parser.add_argument("--mesh_dir", type=str, default="./visual_datas/mesh_debug_samples/", help='save dir')
    parser.add_argument('--motion_list', default="002103", nargs="+", type=str, help="motion name list")
    parser.add_argument("--mode", type=str, default="sequence")
    parser.add_argument("--image_quality", type=str, default="med")
    # for mode=sequence
    parser.add_argument("--vis_frame_num", type=int, default=8)
    # for mode=frame
    parser.add_argument("--down_sample", type=int, default=1)
    # for mode=keyframe
    parser.add_argument("--keyframe_id", default=-1, nargs="+", type=int)
    parser.add_argument("--only_keyframe", action="store_true", default=False)

    parser.add_argument("--start_idx", type=int, default=0)
    # parser.add_argument("--vis_num", type=int, default=200)


    parser.add_argument("--color", type=str, default="Blues")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--on_floor", action="store_true", default=False)
    parser.add_argument("--disable_floor", action="store_true", default=False)
    return parser.parse_args()


def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def go():
    args = get_args()

    device = args.device
    if device != "cpu":
        device = torch.device(f"cuda:{device}")

    # print(os.listdir(os.getcwd()))
    # exit()
    vis_num = 100

    print("start_idx: %d, vis_num: %d" % (args.start_idx, vis_num))

    fr_test_split = open("./data/HumanML3D_guo/test.txt")

    samples_ids = [x.strip() for x in fr_test_split.readlines()]
    samples_ids = samples_ids[args.start_idx * vis_num:(args.start_idx+1) * vis_num]

    make_dir(args.frame_dir)
    frame_dir = args.frame_dir + "/%d" % args.start_idx

    make_dir(frame_dir)
    make_dir(args.mesh_dir)
    make_dir(args.video_dir)


    file_dir = args.file_dir

    from tqdm import tqdm
    for i, sample_id in tqdm(enumerate(samples_ids)):
        # sample_id = "001567"
        # sample_id = "new_010797_text0_kf29_01"
        # sample_id = "new_010797_text0_kf29_17"
        sample_id = "new_010797_text0_kf29-69_02"

        print(sample_id)
        motion = np.load(osp.join(file_dir, sample_id + ".npy"))
        # motion = np.load(osp.join(file_dir, filename + ".npy"))
        # print('processing: ', filename, motion.shape)

        # time.sleep(1)
        # logfile = 'blender_render.log'
        # open(logfile, 'a').close()
        # old = os.dup(sys.stdout.fileno())
        # sys.stdout.flush()
        # os.close(sys.stdout.fileno())
        # fd = os.open(logfile, os.O_WRONLY)

        render_image(motion, name=sample_id, frame_dir=frame_dir, video_dir=args.video_dir,
                     sequence_dir=args.sequence_dir, mesh_dir=args.mesh_dir, mode=args.mode,
                     vis_frame_num=args.vis_frame_num, color=args.color, device=device,
                     on_floor=args.on_floor, down_sample=args.down_sample, disable_floor=args.disable_floor,
                     keyframe_id=args.keyframe_id, only_keyframe=args.only_keyframe, image_quality=args.image_quality)
        # os.close(fd)
        # os.dup(old)
        # os.close(old)
        # time.sleep(1)
        break
        # if i > 0:
        #     break


if __name__ == "__main__":
    go()