import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

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
    TexturesUV,
    TexturesVertex
)

# plot grid images
from plot_image_grid import image_grid
from scipy.spatial.transform import Rotation 


class Renderer():
    def __init__(self, image_size = (720,1280)):
        # Setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        
        self.image_size = image_size
        # initial cameras, raster, lights settings
        focals = torch.tensor([[914.4831543, 913.00628662]], device = self.device)
        principals = torch.tensor([[645.47546387, 367.93243408]], device = self.device)
        K = torch.tensor([[[914.4831543 ,   0.        , 645.47546387,      0.],
                          [  0.        , 913.00628662, 367.93243408,      0.],
                          [  0.        ,   0.        ,   0.        ,      1.], 
                          [  0.        ,   0.        ,   1.        ,      0.]]], device = self.device) 

        #K = K/1000        
        extrinsic = extrinsic = np.array([[-0.91087912, -0.40173757, -0.09437244,  0.1],
                      [-0.13151312,  0.49935385, -0.85635859,  0.1],
                      [ 0.39115666, -0.76762794, -0.50768475,  0.374397 ],
                      [ 0.        ,  0.        ,  0.        ,  1.        ]])

        camera_pose = extrinsic
        #camera_pose[:,:2] = -camera_pose[:,:2]
        print('camera euler angle: ', Rotation.from_matrix(camera_pose[:3, :3]).as_euler('xyz', degrees=True)) 
       
        # test rotation matrix
        

        R = torch.tensor([camera_pose[:3, :3]], device = self.device)
        t = torch.tensor([camera_pose[:3, 3]], device = self.device)

        # only one camera array
        self.cameras = PerspectiveCameras(in_ndc = False, device=self.device, R=R, T=t, K = K, image_size = torch.tensor([self.image_size])) 

        self.raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])

    def load_meshes(self, files, textures = True):
        # files : A list
        self.meshes = load_objs_as_meshes(files, device = self.device, load_textures=textures)

    def update_light_position(self, position):
        self.lights.location = torch.Tensor([position], device = self.device)



    def build_color_renderer(self):
        self.color_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights
            )
        )
    
    def render(self):
        return self.color_renderer(self.meshes)



ren = Renderer()
obj_filename = '/hri/localdisk/yjin/intro_ros_ws/src/intro_object_models/models/ycb-video/025_mug/textured_simple.obj' 
ren.load_meshes([obj_filename])

ren.build_color_renderer()
images = ren.render()
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., 3].cpu().numpy())
plt.axis("off");
plt.show()
