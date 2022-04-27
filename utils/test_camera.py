# Developed by Zijie Tan,Zepeng Xiao,Shiyu Xiu
# In this file, we tested and figured out a way to use parameters stored in meta data to 
# render generated volumes exactly same pose as input image view.
# However, it is not directly used in current implementation.



import pstats
from django.shortcuts import render
import torch
import matplotlib.pyplot as plt
from pytorch3d.datasets import R2N2
from pytorch3d.datasets import r2n2
import math
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher
)
from pytorch3d.structures import Volumes



# from utils.binvox_rw import read_as_3d_array, read_as_coord_array
# import utils.camera as camera
#from utils.binvox_rw import read_as_3d_array, read_as_coord_array

from binvox_rw import read_as_3d_array, read_as_coord_array
import pytorch3d.renderer

#import utils.camera as camera
import camera as camera

import numpy as np
import cv2 as cv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def compute_extrinsic_matrix(azimuth, elevation, distance):
    """
    Compute 4x4 extrinsic matrix that converts from homogenous world coordinates
    to homogenous camera coordinates. We assume that the camera is looking at the
    origin.
    Inputs:
    - azimuth: Rotation about the z-axis, in degrees
    - elevation: Rotation above the xy-plane, in degrees
    - distance: Distance from the origin
    Returns:
    - FloatTensor of shape (4, 4)
    """
    azimuth, elevation, distance = (float(azimuth), float(elevation), float(distance))
    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor([[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]])
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -R_obj2cam.mm(cam_location)
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # For some reason I cannot fathom, when Blender loads a .obj file it rotates
    # the model 90 degrees about the x axis. To compensate for this quirk we roll
    # that rotation into the extrinsic matrix 
    
    theta_X = math.pi/2
    theta_Y = math.pi/2
    theta_Z = 0
    rot_X = torch.tensor([[1, 0, 0, 0], [0, np.cos(theta_X), -1*np.sin(theta_X), 0], [0, np.sin(theta_X), np.cos(theta_X), 0], [0, 0, 0, 1]])
    rot_Y = torch.tensor([[np.cos(theta_Y), 0, np.sin(theta_Y), 0], [0, 1, 0, 0], [-1*np.sin(theta_Y), 0, np.cos(theta_Y), 0], [0, 0, 0, 1]])
    rot_Z = torch.tensor([[np.cos(theta_Z), -1*np.sin(theta_Z), 0, 0], [np.sin(theta_Z), np.cos(theta_Z), 0 , 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    
    rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    #RT = RT.mm(rot.to(RT))
    RT = RT.mm(rot_X.to(RT))
    RT = RT.mm(rot_Y.to(RT))
    #RT = RT.mm(rot_Y.to(RT))

    return RT
    
def main():
    model_views = [1]
    with open("../../Pix2Vox/ShapeNetRendering/02691156/1a04e3eab45ca15dd86060f189eb133/rendering/rendering_metadata.txt") as fm:
        metadata_lines = fm.readlines()
    for i in model_views:
        azim, elev, yaw, dist_ratio, fov = [
            float(v) for v in metadata_lines[i].strip().split(" ")
        ]
    print([azim, elev, yaw, dist_ratio, fov])
    RT = compute_extrinsic_matrix(azim, elev, dist_ratio)
    R, T = camera.compute_camera_calibration(RT)
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist_ratio, elev=elev, azim=azim)

    # Intrinsic matrix extracted from the Blender with slight modification to work with
    # PyTorch3D world space. Taken from meshrcnn codebase:
    # https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py
    K = torch.tensor(
        [
            [2.1875, 0.0, 0.0, 0.0],
            [0.0, 2.1875, 0.0, 0.0],
            [0.0, 0.0, -1.002002, -0.2002002],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    """
    K = torch.tensor([
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ])
    """
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
        K=Ks,
        fov=fov, 
        aspect_ratio=dist_ratio,
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
    with open("../../Pix2Vox/ShapeNetVox32/02691156/1a04e3eab45ca15dd86060f189eb133/model.binvox", "rb") as fp:

    #with open("C:\\Users\\MK12_\\Source\\Pix2Vox\\ShapeNetVox32\\02691156\\1a04e3eab45ca15dd86060f189eb133\\model.binvox", "rb") as fp:
        array_3d = read_as_3d_array(fp)
        array_3d = torch.tensor(array_3d.data, dtype=torch.float32)

        colors = torch.zeros(*array_3d.shape)
        colors[array_3d==1] = 1

        b = torch.sum(colors)

        volume_size = 32
        colors = colors.expand(1, 3, *array_3d.shape)
        dens = array_3d.expand(1, 1, *array_3d.shape)
        volume = Volumes(
            densities=dens, 
            features=colors,
            voxel_size=(volume_extent_world/volume_size)/3.5
        )
        rendered_images, rendered_silhouettes = vox_renderer(cameras=fovCameras, volumes=volume)
        rendered_images, rendered_silhouettes = vox_renderer(cameras=fovCameras, volumes=volume)[0].split([3,1],dim=-1)        
        # print("rendered_img:", rendered_images[0])
        a = torch.sum(rendered_images[0])
        # print("rendered_silhouettes", rendered_silhouettes[0])
        
        clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()

        #print("r layer:\n",rendered_images[0][100,80:120,0])
        #print("g layer:\n",torch.sum(rendered_images[0][:,:,1]))
        #print("b layer:\n",torch.sum(rendered_images[0][:,:,2]))
        print(rendered_images.shape)
        plt.imshow(rendered_images[0][:,:,:3]) 
        
        #plt.imshow(cv.flip(np.array(rendered_images[0][:,:,:3]),0)) 
        plt.show()
        
        print(rendered_silhouettes.shape)
        
        plt.imshow(rendered_silhouettes[0])
        
        
        """
        print("sum:\n", torch.sum(rendered_images[0]))
        print("image shape:",rendered_images[0].shape)
        print("end")
        print("rendered_silhouettes:",len(rendered_silhouettes))
        
        
        print("rendered_shape:",rendered_silhouettes[1][0])
        
        #plt.imshow(clamp_and_detach(rendered_silhouettes[...,0])) 
        plt.imshow(rendered_silhouettes[1][0])
        
        print('0:\n',rendered_silhouettes[0].shape)
        print('1:\n',rendered_silhouettes[1].shape)
        print('2:\n',rendered_silhouettes[2].shape)
        print('3:\n',rendered_silhouettes[3].shape)
        """
        
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

if __name__ == "__main__":
    main()