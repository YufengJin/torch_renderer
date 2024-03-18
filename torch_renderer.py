import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
from scipy.spatial.transform import Rotation
# conversion from opencv to pytorch3d
from pytorch3d.utils import cameras_from_opencv_projection
# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    BlendParams,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.transforms import (
    quaternion_to_matrix,
    quaternion_apply,
    matrix_to_quaternion
)

# TODO clean up the initial parameter for a convenient inheritant
class DifferentiableRenderer:
    def __init__(self, K, image_size, device='cuda:0'):
        assert isinstance(K, torch.Tensor), "[Error] DifferentiableRenderer.__init__: K must be a torch.Tensor"
        assert isinstance(K, torch.Tensor), "[Error] DifferentiableRenderer.__init__: image_size must be a torch.Tensor"

        if len(K.shape) == 2:
            K = K.unsqueeze(0)

        self._K = K
        
        if not isinstance(image_size, tuple):
            print("ERROR: DifferentiableRenderer.__init__: image_size must be a tuple, e.g, (720, 1280)")
            raise 

        self._image_size = image_size
        self._device = device


        # initialize PerspectiveCamera
        self._initialize_perspective_cameras()


    def _initialize_perspective_cameras(self):
        camera_matrix = self._K
       
        focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
        principal_point = camera_matrix[:, :2, 2]

        self._cameras = PerspectiveCameras(focal_length=focal_length,
                                           principal_point=principal_point,
                                           device=self._device,
                                           in_ndc=False,
                                           image_size=torch.tensor([self._image_size]))

    @staticmethod
    def _camera_pose_from_opencv_to_pytorch(R, tvec):
        # TODO boardcasting the R,T
        R_pytorch3d = R.clone().permute(0, 2, 1)
        T_pytorch3d = tvec.clone()
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1
        return R_pytorch3d, T_pytorch3d


class DepthRender(DifferentiableRenderer):
    def __init__(self, K, image_size, faces_per_pixel=1, device="cuda:0"):
        super().__init__(K, image_size, device)
        print("INFO: Initializing DepthRender ...")
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        # customized setting for depth and silhouette render
        raster_settings = RasterizationSettings(
            image_size=self._image_size,
            blur_radius= 0., 
            faces_per_pixel=faces_per_pixel,
            #perspective_correct = False                   # activate if backprop does not work
        )

        self._rasterizer = MeshRasterizer(
                cameras = self._cameras,
                raster_settings = raster_settings
            )

        self._silhouette_renderer = MeshRenderer(
            rasterizer= self._rasterizer, 
            shader=SoftSilhouetteShader(
                # blend param deactivated 
                # blend_params = blend_params              
            )
        )

    def render(self, meshes, R, tvec, return_silhouette=False):
        assert isinstance(meshes, Meshes), "[Error] PointRender.render, meshes must be pytorch3d.structures.Meshes"
        Rs, ts = self._camera_pose_from_opencv_to_pytorch(R, tvec)
        zbuf = self._rasterizer(meshes, R=Rs, T=ts).zbuf[..., 0]      # get closest faces to camera for each pixels
        depths = torch.relu(zbuf)

        if not return_silhouette:
            return depths

        # silhouettes is not binay mask 
        silhouettes = self._silhouette_renderer(meshes, R=Rs, T=ts)
        return depths, silhouettes[..., 3]


class ColorRender(DifferentiableRenderer):
    def __init__(self, K, image_size, blur_radius=0., faces_per_pixel=1, device="cuda:0"):
        super().__init__(K, image_size, device)
        print("INFO: Initializing ColorRender ...")

        blend_params = BlendParams(sigma = blur_radius, gamma = blur_radius) if blur_radius != 0. else 0.

        # TODO enable dynamic light setting
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # customized setting for depth and silhouette render
        raster_settings = RasterizationSettings(
            image_size=self._image_size,
            blur_radius= blend_params,
            faces_per_pixel=faces_per_pixel,
        )

        rasterizer = MeshRasterizer(
                cameras = self._cameras,
                raster_settings = raster_settings
            )

        self._phong_renderer = MeshRenderer(
            rasterizer= rasterizer,
            shader=SoftPhongShader(
                device = self._device,
                cameras= self._cameras,
                lights = lights
            )
        )
                     
    def render(self, meshes, R, tvec):
        assert isinstance(meshes, Meshes), "[Error] PointRender.render, meshes must be pytorch3d.structures.Meshes"
        Rs, ts = self._camera_pose_from_opencv_to_pytorch(R, tvec)
        images = self._phong_renderer(meshes, R=Rs, T=ts)
        return images[..., :3]


# TODO PointRenders not tested  
class AlphaPointRender(DifferentiableRenderer):
    def __init__(self, K, image_size, radius=0.003, points_per_pixel=10, background_color = (0, 0, 0), device="cuda:0"):
        super().__init__(K, image_size, device=device)
        print("INFO: Initializing AlphaPointRender ...")
        # customized setting for depth and silhouette render
        raster_settings = PointsRasterizationSettings(
            image_size=self._image_size, 
            radius = radius,
            points_per_pixel = points_per_pixel
        )

        rasterizer = PointsRasterizer(cameras=self._cameras, raster_settings=raster_settings)
        self._renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=background_color)
        )

    def render(self, points, R, tvec):
        assert isinstance(points, Pointclouds), "[Error] PointRender.render, points must be pytorch3d.structures.Pointclouds"
        Rs, ts = self._camera_pose_from_opencv_to_pytorch(R, tvec)
        images = self._renderer(points, R=Rs, T=Ts)
        return images[...,:3] 

class NormPointRender(DifferentiableRenderer):
    def __init__(self, K, image_size, radius=0.003, points_per_pixel=10, background_color = (0, 0, 0), device="cuda:0"):
        super().__init__(K, image_size, device=device)
        print("INFO: Initializing NormPointRender ...")
        # customized setting for depth and silhouette render
        raster_settings = PointsRasterizationSettings(
            image_size=self._image_size, 
            radius = radius,
            points_per_pixel = points_per_pixel
        )

        rasterizer = PointsRasterizer(cameras=self._cameras, raster_settings=raster_settings)
        self._renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=NormWeightedCompositor(background_color=background_color)
        )

    def render(self, points, R, tvec):
        assert isinstance(points, Pointclouds), "[Error] PointRender.render, points must be pytorch3d.structures.Pointclouds"
        Rs, ts = self._camera_pose_from_opencv_to_pytorch(R, tvec)
        images = self._renderer(points, R=Rs, T=Ts)
        return images[...,:3] 

class PulsarPointRender(DifferentiableRenderer):
    def __init__(self, K, image_size, radius=0.003, points_per_pixel=10, device="cuda:0"):
        super().__init__(K, image_size, device=device)
        print("INFO: Initializing PulsarPointRender ...")
        # customized setting for depth and silhouette render
        raster_settings = PointsRasterizationSettings(
            image_size=self._image_size, 
            radius = radius,
            points_per_pixel = points_per_pixel
        )
        
        self._renderer =PulsarPointsRenderer(
            rasterizer=PointsRasterizer(cameras=self._cameras, raster_settings=raster_settings),
            n_channels=4
        ).to(device)

    def render(self, points, R, tvec):
        assert isinstance(points, Pointclouds), "[Error] PointRender.render, points must be pytorch3d.structures.Pointclouds"
        Rs, ts = self._camera_pose_from_opencv_to_pytorch(R, tvec)
        images = self._renderer(points, R=Rs, T=Ts, gamma=(1e-4,),
                  bg_col=torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32, device=device))
        return images[...,:3] 
