import pstats
import torch
import camera
from pytorch3d.datasets import R2N2
from pytorch3d.datasets import r2n2

model_views = [0]
device = 'cpu'

with open("./ShapeNetRendering/02691156/1a04e3eab45ca15dd86060f189eb133/rendering/rendering_metadata.txt") as fm:
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
        [0.0, 0.0, -1.002002, -0.2002002],
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
    device=device
).to(device)
pass
