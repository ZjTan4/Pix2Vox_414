# Helper functions modified from pytorch3D by Zijie Tan, Zepeng Xiao, Shiyu Xiu
# Used to compute extrinsic camera parameters from the parameters sphere gemometry
# 

import math
from typing import Dict, List

import numpy as np
import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.datasets.utils import collate_batched_meshes
from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import Transform3d

# Empirical min and max over the dataset from meshrcnn.
# https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py#L9
SHAPENET_MIN_ZMIN = 0.67
SHAPENET_MAX_ZMAX = 0.92
# Threshold for cubify from meshrcnn:
# https://github.com/facebookresearch/meshrcnn/blob/main/configs/shapenet/voxmesh_R50.yaml#L11
CUBIFY_THRESH = 0.2

# Default values of rotation, translation and intrinsic matrices for BlenderCamera.
r = np.expand_dims(np.eye(3), axis=0)  # (1, 3, 3)
t = np.expand_dims(np.zeros(3), axis=0)  # (1, 3)
k = np.expand_dims(np.eye(4), axis=0)  # (1, 4, 4)

def compute_extrinsic_matrix(
    azimuth: float, elevation: float, distance: float
):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py#L96
    Compute 4x4 extrinsic matrix that converts from homogeneous world coordinates
    to homogeneous camera coordinates. We assume that the camera is looking at the
    origin.
    Used in R2N2 Dataset when computing calibration matrices.
    Args:
        azimuth: Rotation about the z-axis, in degrees.
        elevation: Rotation above the xy-plane, in degrees.
        distance: Distance from the origin.
    Returns:
        FloatTensor of shape (4, 4).
    """
    azimuth, elevation, distance = float(azimuth), float(elevation), float(distance)

    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor(
        [[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]]
    )
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -(R_obj2cam.mm(cam_location))
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # Georgia: For some reason I cannot fathom, when Blender loads a .obj file it
    # rotates the model 90 degrees about the x axis. To compensate for this quirk we
    # roll that rotation into the extrinsic matrix here
    rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    # pyre-fixme[16]: `Tensor` has no attribute `mm`.
    RT = RT.mm(rot.to(RT))

    return RT

def compute_camera_calibration(RT):
    """
    Helper function for calculating rotation and translation matrices from ShapeNet
    to camera transformation and ShapeNet to PyTorch3D transformation.
    Args:
        RT: Extrinsic matrix that performs ShapeNet world view to camera view
            transformation.
    Returns:
        R: Rotation matrix of shape (3, 3).
        T: Translation matrix of shape (3).
    """
    # Transform the mesh vertices from shapenet world to pytorch3d world.
    shapenet_to_pytorch3d = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    RT = torch.transpose(RT, 0, 1).mm(shapenet_to_pytorch3d)  # (4, 4)
    # Extract rotation and translation matrices from RT.
    R = RT[:3, :3]
    T = RT[3, :3]
    return R, T

class BlenderCamera(CamerasBase):  # pragma: no cover
    """
    Camera for rendering objects with calibration matrices from the R2N2 dataset
    (which uses Blender for rendering the views for each model).
    """

    def __init__(self, R=r, T=t, K=k, device: Device = "cpu") -> None:
        """
        Args:
            R: Rotation matrix of shape (N, 3, 3).
            T: Translation matrix of shape (N, 3).
            K: Intrinsic matrix of shape (N, 4, 4).
            device: Device (as str or torch.device).
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(device=device, R=R, T=T, K=K)

    def get_projection_transform(self, **kwargs) -> Transform3d:
        transform = Transform3d(device=self.device)
        transform._matrix = self.K.transpose(1, 2).contiguous()
        return transform

    def is_perspective(self):
        return False

    def in_ndc(self):
        return True
