import pstats
from django.shortcuts import render
import torch
import matplotlib.pyplot as plt
from pytorch3d.datasets import R2N2
from pytorch3d.datasets import r2n2

from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher
)
from pytorch3d.renderer.implicit.utils import ray_bundle_to_ray_points
from pytorch3d.structures import Volumes
# from utils.binvox_rw import read_as_3d_array, read_as_coord_array
# import utils.camera as camera
from binvox_rw import read_as_3d_array, read_as_coord_array
import camera



model_views = [7]

with open("../Pix2Vox/ShapeNetRendering/02691156/1a04e3eab45ca15dd86060f189eb133/rendering/rendering_metadata.txt") as fm:
    metadata_lines = fm.readlines()
for i in model_views:
    azim, elev, yaw, dist_ratio, fov = [
        float(v) for v in metadata_lines[i].strip().split(" ")
    ]
RT = r2n2.utils.compute_extrinsic_matrix(azim, elev, dist_ratio)
R, T = camera.compute_camera_calibration(RT)

# Intrinsic matrix extracted from the Blender with slight modification to work with
# PyTorch3D world space. Taken from meshrcnn codebase:
# https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py
K = torch.tensor(
    [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -2.02002, -0.2002002],
        [0.0, 0.0, 1.0, 0.0],
    ]
)
Rs = torch.stack([R])
Ts = torch.stack([T])
Ks = K.expand(1, 4, 4)

blenderCamera = r2n2.utils.BlenderCamera(
    R=Rs, 
    T=Ts, 
    K=Ks, 
)

fovCameras = FoVPerspectiveCameras(
    R=Rs, 
    T=Ts,
    # K=Ks,
    # fov=fov, 
    # aspect_ratio=dist_ratio,
)

# volumetric renderer
# render_size = 576
render_size = 224
volume_extent_world = 1.5

raysampler = NDCMultinomialRaysampler(
    image_width=render_size, 
    image_height=render_size,
    n_pts_per_ray=50, 
    min_depth=0.1,
    max_depth=volume_extent_world
)
raymarcher = EmissionAbsorptionRaymarcher()

vox_renderer = VolumeRenderer(
    raysampler=raysampler, 
    raymarcher=raymarcher
)

with open("C:\\Users\\MK12_\\Source\\Pix2Vox\\ShapeNetVox32\\02691156\\1a04e3eab45ca15dd86060f189eb133\\model.binvox", "rb") as fp:
    array_3d = read_as_3d_array(fp)
    array_3d = torch.tensor(array_3d.data, dtype=torch.float32)

    colors = torch.zeros(*array_3d.shape)
    colors[array_3d==1] = 1

    # b = torch.sum(colors)

    volume_size = 32
    colors = colors.expand(1, 3, *array_3d.shape)
    dens = array_3d.expand(1, 1, *array_3d.shape)
    volume = Volumes(
        densities=dens, 
        features=colors,
        voxel_size=(volume_extent_world/volume_size) / 2
    )
    rendered_images, rendered_silhouettes = vox_renderer(cameras=fovCameras, volumes=volume)[0].split([3, 1], dim=-1)
    # print("rendered_img:", rendered_images[0])
    # a = torch.sum(rendered_images[0])
    # print("rendered_silhouettes", rendered_silhouettes[0])
    plt.imshow(rendered_images[0])

    plt.show()
    print("end")


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.
    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
