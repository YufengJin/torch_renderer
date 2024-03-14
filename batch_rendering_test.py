#############################################################################
# Test Code for batch rendering
# 
#
#############################################################################
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import pyrender
import trimesh
import pickle
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
from scipy.spatial.transform import Rotation
# conversion from opencv to pytorch3d
from pytorch3d.utils import cameras_from_opencv_projection
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,

)
from pytorch3d.transforms import (
    quaternion_to_matrix,
    quaternion_apply,
    matrix_to_quaternion
)
from visualizer import VisPyrender

# activate GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def cameras_from_opencv_projection_no_ndc(R, tvec, camera_matrix, image_size, device):
    # from opencv projection to pytorch3d

    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]
    print('focal_length, principal point shape: ', focal_length.shape, principal_point.shape)
    # # Retype the image_size correctly and flip to width, height.
    # image_size_wh = image_size.to(R).flip(dims=(1,))

    # # Get the PyTorch3D focal length and principal point.
    # focal_pytorch3d = focal_length / (0.5 * image_size_wh)
    # p0_pytorch3d = -(principal_point / (0.5 * image_size_wh) - 1)

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1
    return PerspectiveCameras(focal_length=focal_length,
                              principal_point=principal_point,
                              R=R_pytorch3d,
                              T=T_pytorch3d,
                              device=device,
                              in_ndc=False,
                              image_size=image_size), R_pytorch3d, T_pytorch3d


def look_at_view_transform_np(
    distance,
    elevation,
    azimuth,
    inplane_rol = 180.,
    degree=True):

    # Camera Extrinsic from distance, elecvation, and azimuth

    dist, elev, azim, rol = distance, elevation, azimuth, inplane_rol

    if degree:
        elev = math.pi / 180.0 * elev
        rol  = math.pi / 180.0 * rol
        azim = math.pi / 180.0 * azim

    x = dist * math.cos(elev) * math.sin(azim)
    y = dist * math.sin(elev) * math.sin(azim)
    z = dist * math.cos(azim)

    #translation = (x, y, z)
    transform = np.eye(4)

    rotMat = np.zeros(9)
    rotMat[0] = np.cos(rol) * np.cos(azim) - np.sin(rol) * np.cos(elev) * np.sin(azim)
    rotMat[1] = np.sin(rol) * np.cos(azim) + np.cos(rol) * np.cos(elev) * np.sin(azim)
    rotMat[2] = np.sin(elev) * np.sin(azim)
    rotMat[3] = -np.cos(rol) * np.sin(azim) - np.sin(rol) * np.cos(elev) * np.cos(azim)
    rotMat[4] = -np.sin(rol) * np.sin(azim) + np.cos(rol) * np.cos(elev) * np.cos(azim)
    rotMat[5] = np.sin(elev) * np.cos(azim)
    rotMat[6] = np.sin(rol) * np.sin(elev)
    rotMat[7] = -np.cos(rol) * np.sin(elev)
    rotMat[8] = np.cos(elev)

    rotMat = rotMat.reshape(3,3)
    trans = np.array([0,0,dist])
    translation = rotMat@trans.T
    transform[:3,:3] = rotMat.reshape(3,3)


    transform[:3,3] = translation.T
    return transform


class VisTorch3D:
    def __init__(self, width=1280, height=720, scale_factor=1):
        # device 
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            print(f"VisTorch running on GPU, device: {device}")
        else:
            device = torch.device("cpu")
            print("VisTorch running on CPU")
        self._device = device
    
        # private params
        self._cameras = None
    
        # scaling 
        self._cindex = scale_factor
        width = width // scale_factor
        height = height // scale_factor
    
        self._image_size = image_size = (height, width)
        
        # raster setting
        self._raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        self._blend_params = BlendParams(sigma = 1e-4, gamma = 1e-4)

        self._lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

        #self._silhouette_raster_settings = RasterizationSettings(
        #    image_size=image_size,
        #    blur_radius=np.log(1./1e-4 -1.) * self._blend_params.sigma,
        #    # TODO figure out he meaning of arguments
        #    faces_per_pixel=10,
        #)

        self._silhouette_raster_settings = self._raster_settings
        
    
    def rendererSetup(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self._raster_settings
            ),
            shader=SoftPhongShader(
                device=self._device,
                cameras=self.cameras,
                lights=self._lights
            )
        )
        
        
        # silhouette renderer
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self._silhouette_raster_settings
            ),
            shader=SoftSilhouetteShader(
                #blend_params = blend_params
            )
        )
        
        self.fragments = rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self._silhouette_raster_settings
            )
        print("Color renderer, Silhouette renderer and depth fragement renderer created.")

    def set_reference(self, depth, mask):
        assert len(depth.shape) == len(mask.shape) == 2, "SET REFERENCE ERROR, depth and mask must be 2 dim"
        self.depth_ref = torch.from_numpy(depth).float().unsqueeze(0).to(self._device)
        self.mask_ref = torch.from_numpy(mask).float().unsqueeze(0).to(self._device)

    def set_IntrinsicsCameras(self, intrinsicMats):
        # intrinsicMats is a list of intrinsic Matrices
        if self._cameras is not None:
            return
        else:
            camera_matrix = np.array(intrinsicMats, dtype=np.float32)
            camera_matrix[:,:3,:3] *= 1 / self._cindex 
             
            # numpy to tensor
            camera_matrix = torch.from_numpy(camera_matrix).float().to(self._device)
            focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
            principal_point = camera_matrix[:, :2, 2]
    
            image_size = torch.tensor([self._image_size], dtype=torch.float32).to(self._device)
            image_size = image_size.repeat(len(focal_length), 1)
    
            print(f"Perspective Cameras repeated {len(focal_length)} in {self._device}")
            
            self.cameras = PerspectiveCameras(focal_length=focal_length,
                                               principal_point=principal_point,
                                               device=self._device,
                                               in_ndc=False,
                                               image_size=image_size)

            self.rendererSetup()

    @staticmethod
    def camera_pose_from_opencv_to_pytorch(R, tvec):
        assert isinstance(R, torch.Tensor) and isinstance(tvec, torch.Tensor), "ERROR: R, T are not tensor.Tensor"
        assert len(R.size()) == 3 and len(tvec.size()) == 2, "ERROR: Shape of R, T are not valid"
        R_pytorch3d = R.clone().permute(0, 2, 1)
        T_pytorch3d = tvec.clone()
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1
        return R_pytorch3d, T_pytorch3d

    def render_rgb(self, meshes, Rs, Ts):
        assert len(meshes) == len(Rs) == len(Ts), "ERROR: Length of meshes, R and T must be identical for batch rendering"
        tic = time.time() 
        #meshes = meshes.to(self._device)
        Rs, Ts = Rs.to(self._device), Ts.to(self._device)
        Rs, Ts = self.camera_pose_from_opencv_to_pytorch(Rs, Ts)
        print(f"VisTorch: Data Transfer from cpu to GPU takes: {time.time()-tic:.6f} s")

        print(f"Mesh, Rs and Ts are on device : {meshes.device} {Rs.device} {Ts.device} respectively")
        colors = self.renderer(meshes, R = Rs, T=Ts)
        
        # convert to numpy
        colors = colors[...,:3].cpu().numpy()
        return colors

    def _calc_confidence(self, rendered_depthes, w_dist = 0.5, clip_tolerance = 0.05)
        # TODO batch confidence matrics
        pass
        
    #@profile
    def render_depth(self, meshes, Rs, Ts):
        assert len(meshes) == len(Rs) == len(Ts), "ERROR: Length of meshes, R and T must be identical for batch rendering"
        tic = time.time() 
        #meshes = meshes.to(self._device)
        Rs, Ts = Rs.to(self._device), Ts.to(self._device)
        Rs, Ts = self.camera_pose_from_opencv_to_pytorch(Rs, Ts)
        print(f"VisTorch: Data Transfer from cpu to GPU takes: {time.time()-tic:.6f} s")

        #print(f"Mesh, Rs and Ts are on device : {meshes.device} {Rs.device} {Ts.device} respectively")

        # zbuf giving the z-coordinates of the nearest faces at each pixels in world coordinate, sorted in acending z-order
        depthes = self.fragments(meshes, R = Rs, T=Ts).zbuf
        # convert to numpy

        depthes = depthes[..., 0].cpu().numpy()
        depthes[depthes == -1.] = 0.
        # TODO depth could do some erosion
        return depthes

# camera intrinsic, image size
cameraIntrinsic = np.array([917.700439453125, 0.0, 646.2421875, 0.0, 0.0, 915.8268432617188, 367.7502136230469, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.], dtype='float32').reshape(4,4)
image_size = (720,1280)
scale_factor = 1 
# camera extrinsic
dist = 0.7
elev = 120.
azim = 0.

cameraExtrinsic = look_at_view_transform_np(dist, elev, azim)
# rostf likely
cameraExtrinsic[:3, 1:3] = -cameraExtrinsic[:3, 1:3]
cameraExtrinsic = np.linalg.inv(cameraExtrinsic.copy())

# initial pyrender
visPyrender = VisPyrender(scale_factor=scale_factor)
visPyrender.set_IntrinsicsCamera(cameraIntrinsic)

# initial VisTorch3d
visTorch = VisTorch3D(scale_factor=scale_factor)
visTorch.set_IntrinsicsCameras([cameraIntrinsic])


# load object pyrender and pytorch3d
objId = 25 
object_p = f"/hri/localdisk/yjin/intro_ros_ws/src/intro_object_models/models/all_objects/{objId:03d}/colored_points.obj"
obj_trimesh = trimesh.load(object_p)
mesh_pyrender = pyrender.Mesh.from_trimesh(obj_trimesh)

# put mesh into GPU in advances
device = torch.device("cuda:0")
meshes_torch3d = load_objs_as_meshes([object_p], device=device)

#@profile
def run_pyrender(mesh, camera_pose):
    object_pose = np.eye(4)
    return visPyrender.quick_depth_render(mesh, object_pose, camera_pose)
#@profile
def run_torch3d(mesh, camera_pose, N=1):
    camera_pose = torch.from_numpy(camera_pose).float()
    R = camera_pose[:3, :3][None]
    T = camera_pose[:3, 3][None]

    # extend Meshes, R and t tensors
    meshes = meshes_torch3d.extend(N)
    Rs = R.repeat(N,1,1); Ts = T.repeat(N,1)
    return visTorch.render_depth(meshes, Rs, Ts)
 
def run_torch_confience(mesh, camera_pose, N=1):
    # render a reference depth via pyrender
    object_pose = np.eye(4)
    object_pose[:3, :] += np.random.randn(3,4)* 0.01
    depth_ref = visPyrender.quick_depth_render(mesh, object_pose, camera_pose)
    
    mask_ref = (depth_ref != 0.)

    visTorch.set_reference(depth_ref, mask_ref)
    

if __name__ ==  "__main__": 

    batch_size = 120
    tic0 = time.time()
    
    for _ in range(batch_size):
        pyrenderDepth = run_pyrender(mesh_pyrender, cameraExtrinsic)
    
    tic = time.time()
    msg = f"Render {batch_size} depth image with pyrender takes {tic-tic0:.6f} s"
    print(msg)
    
    tic0 = time.time()
    torchDepthes = run_torch3d(meshes_torch3d, cameraExtrinsic, N=batch_size)
    
    tic = time.time()
    msg = f"Render {batch_size} depth image with Pytorch3D takes {tic-tic0:.6f} s"
    print(msg)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1); plt.imshow(pyrenderDepth);plt.axis("off"); plt.title('Pyrender'); 
    plt.subplot(2,2,2); plt.imshow(torchDepthes[0]);plt.axis("off");plt.title('Pytorch3D')
    plt.subplot(2,2,3); plt.imshow(torchDepthes[0] - pyrenderDepth);plt.axis("off");plt.title('Diff'); plt.colorbar()
    
    plt.show()
    
    
    

