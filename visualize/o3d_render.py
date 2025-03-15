import os
import os.path as osp

import time
from tqdm import tqdm
import numpy as np
import torch
import imageio
import open3d as o3d

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector

from utils.rotation2xyz import Rotation2xyz
from utils.simplify_loc2rot import joints2smpl

#
# def render(motions, outdir='test_vis', name=None, output_type="video"):
#     frames, njoints, nfeats = motions.shape
#     MINS = motions.min(axis=0).min(axis=0)
#     MAXS = motions.max(axis=0).max(axis=0)
#
#     height_offset = MINS[1]
#     motions[:, :, 1] -= height_offset
#     trajec = motions[:, 0, [0, 2]]
#
#     j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
#     rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
#     faces = rot2xyz.smpl_model.faces
#
#     if not os.path.exists(outdir + name + '.pt'):
#         print(f'Running SMPLify, it may take a few minutes.')
#         motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]
#         vertices = rot2xyz(motion_tensor.clone(), mask=None,
#                            pose_rep='rot6d', translation=True, glob=True,
#                            jointstype='vertices',
#                            vertstrans=True)
#
#         torch.save(vertices, osp.join(outdir, name + '.pt'))
#     else:
#         vertices = torch.load(osp.join(outdir, name + '.pt'))
#
#     frames = vertices.shape[3]  # shape: 1, nb_frames, 3, nb_joints
#     print(vertices.shape)
#     MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
#     MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
#     # vertices[:,:,1,:] -= MINS[1] + 1e-5
#
#     minx = MINS[0] - 0.5
#     maxx = MAXS[0] + 0.5
#     minz = MINS[2] - 0.5
#     maxz = MAXS[2] + 0.5
#     polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
#     polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)
#
#     vid = []
#     for i in tqdm(range(frames)):
#         mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)
#
#         base_color = (0.11, 0.53, 0.8, 0.5)
#         ## OPAQUE rendering without alpha
#         ## BLEND rendering consider alpha
#         material = pyrender.MetallicRoughnessMaterial(
#             metallicFactor=0.7,
#             alphaMode='OPAQUE',
#             baseColorFactor=base_color
#         )
#
#         mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
#
#         polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
#         polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)
#
#         bg_color = [1, 1, 1, 0.8]
#         scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
#
#         sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]
#
#         camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
#
#         light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=300)
#
#         scene.add(mesh)
#
#         c = np.pi / 2
#
#         # scene.add(polygon_render, pose=np.array([[1, 0, 0, 0],
#
#         #                                          [0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],
#
#         #                                          [0, np.sin(c), np.cos(c), 0],
#
#         #                                          [0, 0, 0, 1]]))
#
#         light_pose = np.eye(4)
#         light_pose[:3, 3] = [0, -1, 1]
#         scene.add(light, pose=light_pose.copy())
#
#         light_pose[:3, 3] = [0, 1, 1]
#         scene.add(light, pose=light_pose.copy())
#
#         light_pose[:3, 3] = [1, 1, 2]
#         scene.add(light, pose=light_pose.copy())
#
#         c = -np.pi / 6
#
#         scene.add(camera, pose=[[1, 0, 0, (minx + maxx).cpu().numpy() / 2],
#
#                                 [0, np.cos(c), -np.sin(c), 1.5],
#
#                                 [0, np.sin(c), np.cos(c), max(4, minz.cpu().numpy() + (1.5 - MINS[1].cpu().numpy()) * 2,
#                                                               (maxx - minx).cpu().numpy())],
#
#                                 [0, 0, 0, 1]
#                                 ])
#
#         # render scene
#         r = pyrender.OffscreenRenderer(960, 960)
#
#         color, _ = r.render(scene, flags=RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL)
#         # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')
#
#         vid.append(color)
#
#         r.delete()
#
#     out = np.stack(vid, axis=0)
#     imageio.mimsave(osp.join(outdir, name + '.mp4'), out, fps=20)

def create_mesh(vertices, faces, colors=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if colors is not None:
        colors = np.array(colors)
        mesh.paint_uniform_color(colors)
    else:
        r_c = np.random.random(3)
        mesh.paint_uniform_color(r_c)
    return mesh


class Open3DVisualizer:
    def __init__(self, save_img_folder=None, fps=20, name="", enable_axis=False):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Open3D Visualizer", width=960, height=540)

        if enable_axis:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            self.vis.add_geometry(coordinate_frame)

        self.geometry_crt = None
        self.fps = fps

        self.save_img_folder = save_img_folder
        self.video_name = name
        self.frames = []
        # if save_img_folder:
        #     os.makedirs(self.save_img_folder, exist_ok=True)

    def update(self, vertices, faces, color=None, camera_pose=None, floor=None):
        if color is None:
            color = [82.0 / 255, 217.0 / 255, 118.0 / 255]
        mesh = create_mesh(vertices, faces, colors=color)

        self.vis.clear_geometries()
        if floor is not None:
            self.vis.add_geometry(floor)
        self.vis.add_geometry(mesh)

        if camera_pose is not None:
            ctr = self.vis.get_view_control()
            # ctr.change_field_of_view(0)
            camera_params = ctr.convert_to_pinhole_camera_parameters()
            camera_params.extrinsic = camera_pose
            # print(camera_params)
            ctr.convert_from_pinhole_camera_parameters(camera_params)

        self.vis.poll_events()
        self.vis.update_renderer()
        if self.save_img_folder:
            frame = self.vis.capture_screen_float_buffer(False)
            self.frames.append(frame)
            # self.vis.capture_screen_image(
            #     os.path.join(self.save_img_folder, "frame_%04d.png" % self.idx)
            # )
        # time.sleep(1 / self.fps)

    def release(self):
        self.vis.destroy_window()
        # out = np.stack(self.frames, axis=0).astype(np.uint8)
        out = (np.stack(self.frames, axis=0) * 255).astype(np.uint8)

        imageio.mimsave(osp.join(self.save_img_folder, self.video_name + '.mp4'), out, fps=20)

def render_image(motions, outdir='test_vis', name=None, output_type="video"):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces

    if not os.path.exists(outdir + name + '.pt'):
        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]
        vertices = rot2xyz(motion_tensor.clone(), mask=None,
                           pose_rep='rot6d', translation=True, glob=True,
                           jointstype='vertices',
                           vertstrans=True).cpu()

        torch.save(vertices, osp.join(outdir, name + '.pt'))
    else:
        vertices = torch.load(osp.join(outdir, name + '.pt')).cpu()

    vertices = vertices[0]

    frames = vertices.shape[2]  # 6890, 3, frames
    print(vertices.shape)
    MINS = torch.min(torch.min(vertices, axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices, axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= MINS[1] + 1e-5

    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5
    maxz = MAXS[2] + 0.5

    c = 360 / 6

    camera_pose = [[1, 0, 0, (minx + maxx).cpu().numpy() / 2],
                    [0, np.cos(c), -np.sin(c), 1.5],
                    [0, np.sin(c), np.cos(c), max(4, minz.cpu().numpy() + (1.5 - MINS[1].cpu().numpy()) * 2,
                                                  (maxx - minx).cpu().numpy()) + 5],
                    [0, 0, 0, 1]
                    ]

    floor = o3d.geometry.TriangleMesh.create_box(maxx-minx, 0.1, maxz-minz)
    floor.compute_vertex_normals()
    floor.paint_uniform_color([0.1, 0.1, 0.1])
    loc = np.array([-1*(maxx-minx)/2, 0, -1*(maxz-minz)/2])
    floor.translate(loc)

    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=(0, 0, 0))

    vertices[:, 1, :] -= vertices[:, 1, :].min()

    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = Vector3dVector(vertices[:, :, 0].squeeze())
    # mesh.triangles = Vector3iVector(faces)
    # # mesh2 = mesh.filter_smooth_simple(number_of_iterations=5)
    # mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([0.658, 0.214, 0.0114])
    #
    # mesh2 = o3d.geometry.TriangleMesh()
    # mesh2.vertices = Vector3dVector(vertices[:, :, 10].squeeze())
    # mesh2.triangles = Vector3iVector(faces)
    # # mesh2 = mesh.filter_smooth_simple(number_of_iterations=5)
    # mesh2.compute_vertex_normals()
    # mesh2.paint_uniform_color([0.658, 0.214, 0.0114])

    vid = Open3DVisualizer(save_img_folder=outdir, enable_axis=True, name=name)
    for i in range(frames):
        vid.update(vertices[:, :, i].squeeze(), faces, camera_pose=camera_pose, floor=floor)

    vid.release()


    # o3d.visualization.draw_geometries([floor, coords, mesh, mesh2])


if __name__ == "__main__":
    # import open3d as o3d
    #
    # box = o3d.geometry.TriangleMesh.create_box(width=0.3, height=0.4, depth=0.1)
    # box.compute_vertex_normals()
    # box.paint_uniform_color([0.3, 0.5, 1])
    #
    # loc = np.array([0.5, -0.1, 0])
    # box.translate(loc)
    #
    # coords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=(0, 0, 0))


    # box2 = o3d.geometry.TriangleMesh.create_box(width=0.7, height=0.3, depth=0.6)
    # box2.compute_vertex_normals()
    # box2.paint_uniform_color([0.1, 0.5, 0])

    # o3d.visualization.draw_geometries([box])

    # o3d.visualization.draw_geometries([box, coords])
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="./visual_datas/motion_hml3d", help='motion npy file dir', required=False)
    parser.add_argument("--output_dir", type=str, default="./visual_datas/videos/", help='save dir', required=False)
    parser.add_argument('--motion_list', default="002103", nargs="+", type=str,
                        help="motion name list", required=False)
    args = parser.parse_args()

    filename_list = args.motion_list
    file_dir = args.file_dir

    for filename in filename_list:
        motion = np.load(osp.join(file_dir, filename + ".npy"))
        print('processing: ', filename, motion.shape)
        render_image(motion, outdir=args.output_dir, name=filename)

