import os
import shutil

import bpy

from .camera import Camera
from .floor import plot_floor, show_trajectory
from .sampler import get_frameidx
from .scene import setup_scene
from .tools import delete_objs

from mld.render.video import Video


def prune_begin_end(data, perc):
    to_remove = int(len(data) * perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)


def render(npydata, trajectory, path, mode, faces_path, gt=False,
           exact_frame=None, num=8, always_on_floor=False, denoising=True,
           oldrender=True, res="high", accelerator='gpu', device=[0], fps=20):

    if mode == 'video':
        if always_on_floor:
            frames_folder = path.replace(".pkl", "_of_frames")
        else:
            frames_folder = path.replace(".pkl", "_frames")

        if os.path.exists(frames_folder.replace("_frames", ".mp4")) or os.path.exists(frames_folder):
            print(f"pkl is rendered or under rendering {path}")
            return

        os.makedirs(frames_folder, exist_ok=False)

    elif mode == 'sequence':
        path = path.replace('.pkl', '.png')
        img_name, ext = os.path.splitext(path)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}{ext}"
        if os.path.exists(img_path):
            print(f"pkl is rendered or under rendering {img_path}")
            return

    elif mode == 'frame':
        path = path.replace('.pkl', '.png')
        img_name, ext = os.path.splitext(path)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}_{exact_frame}{ext}"
        if os.path.exists(img_path):
            print(f"pkl is rendered or under rendering {img_path}")
            return
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # Setup the scene (lights / render engine / resolution etc)
    setup_scene(res=res, denoising=denoising, oldrender=oldrender, accelerator=accelerator, device=device)

    # remove X% of beginning and end
    # as it is almost always static
    # in this part
    # if mode == "sequence":
    #     perc = 0.2
    #     npydata = prune_begin_end(npydata, perc)

    from .meshes import Meshes
    data = Meshes(npydata, gt=gt, mode=mode, trajectory=trajectory,
                  faces_path=faces_path, always_on_floor=always_on_floor)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    if trajectory is not None:
        show_trajectory(data.trajectory)

    # Create a floor
    plot_floor(data.data, big_plane=False)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode)

    frameidx = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []
    for index, frameidx in enumerate(frameidx):
        if mode == "sequence":
            frac = index / (nframes_to_render - 1)
            mat = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render - 1)

        obj_name = data.load_in_blender(frameidx, mat)
        name = f"{str(index).zfill(4)}"

        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_obj_names.extend(obj_name)
        elif mode == "frame":
            camera.update(data.get_root(frameidx))

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(obj_name)

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    if mode == "video":
        video = Video(frames_folder, fps=fps)
        vid_path = frames_folder.replace("_frames", ".mp4")
        video.save(out_path=vid_path)
        shutil.rmtree(frames_folder)
        print(f"remove tmp fig folder and save video in {vid_path}")

    else:
        print(f"Frame generated at: {img_path}")
